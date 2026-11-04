"""LoopedGPT: a weight-tied GPT-2 block applied K times with input injection.

Local: the layer is the pre-LayerNorm GPT-2 layer of nanoGPT (Karpathy, MIT license,
https://github.com/karpathy/nanoGPT/blob/master/model.py): causal self-attention + GELU
MLP, residual connections, weights initialised N(0, 0.02) with residual projections
scaled by 1/sqrt(2 * n_layer). The looping follows Fan et al. 2024 (arXiv:2409.15647,
Sec. 4.2.1) and Yang et al. 2024 (arXiv:2311.12424, code github.com/Leiay/looped_transformer,
MIT): no positional embeddings (NoPE, position comes from the causal mask), and "input
injection", i.e. the input embedding e is added to the previous iteration's output:
h_t = Block(h_{t-1} + e), h_0 = 0.

Global: `forward(x, loops)` reads each sample's output at its own loop count K_i (used for
training with randomized loops and for curvature measurement); `iterate_loops` yields the
logits after every iteration in one pass (used by evaluation to pick the stopping step).
Attention always runs on PyTorch's MATH SDPA backend: the fused flash / memory-efficient
kernels do not implement double backward, which the Hutchinson penalty needs, and using
one backend everywhere keeps numerics identical across training arms.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from looped_hessian.data import VOCAB_SIZE


@dataclass
class ModelConfig:
    d_model: int = 256  # Fan et al. use embedding dim 256 for all tasks (App. F)
    n_layer: int = 3  # layers inside the shared block (Fan et al. use 3 for addition, App. F Table 2)
    n_head: int = 8  # attention heads (Fan et al. use 8 for addition, App. F Table 2)
    mlp_ratio: int = 4  # GPT-2 MLP expansion
    vocab_size: int = VOCAB_SIZE


class LayerNorm(nn.Module):
    """LayerNorm with a switchable composite-op path that is correct to any derivative order.

    Same math and parameters as nn.LayerNorm (Ba et al. 2016, arXiv:1607.06450): normalise
    over the last dimension with biased variance, eps = 1e-5, then elementwise affine.
    By default the native `F.layer_norm` kernel is used (fast, low memory, and its first and
    second derivatives agree with finite differences, see tests/test_curvature.py). In
    PyTorch 2.15.0 its *third* derivative does not (tests/test_third_order.py), which would
    silently corrupt the gradient of the Hutchinson penalty, so graphs that will be
    differentiated three times are built inside `third_order_safe()`, which switches every
    LayerNorm to the composite formula below.
    """

    composite = False  # class-level switch, toggled only by third_order_safe()

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not LayerNorm.composite:
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        mu = x.mean(-1, keepdim=True)
        # Biased (population) variance, matching nn.LayerNorm.
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        return (x - mu) * torch.rsqrt(var + self.eps) * self.weight + self.bias


@contextmanager
def third_order_safe():
    """Build forward graphs with composite LayerNorm (needed for third-order autograd)."""
    previous = LayerNorm.composite
    LayerNorm.composite = True
    try:
        yield
    finally:
        LayerNorm.composite = previous


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (nanoGPT `CausalSelfAttention`, without dropout)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        # One projection produces queries, keys and values (3 * d_model outputs).
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        # Output projection back into the residual stream.
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        # Reshape to (batch, heads, time, head_dim) as required by scaled_dot_product_attention.
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        # MATH backend: differentiable to any order (see module docstring).
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Merge heads back to (batch, time, d_model) and project.
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    """GPT-2 feed-forward block: Linear -> GELU (exact erf form) -> Linear."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Layer(nn.Module):
    """Pre-LayerNorm transformer layer: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class LoopedGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Token embedding only: no positional embedding (NoPE, Fan et al. Sec. 4.2.1).
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # The shared block: these n_layer layers are reused at every loop iteration.
        self.block = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying between input embedding and output head (nanoGPT / GPT-2 convention).
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        # GPT-2 scaled init for projections that write into the residual stream.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # nanoGPT `_init_weights`: N(0, 0.02) weights, zero biases; LayerNorm keeps (1, 0).
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def step(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """One loop iteration with input injection: Block(h + e)."""
        x = h + e
        for layer in self.block:
            x = layer(x)
        return x

    def head(self, h: torch.Tensor) -> torch.Tensor:
        # Final LayerNorm + tied linear head -> per-position logits over the vocabulary.
        return self.lm_head(self.ln_f(h))

    def forward(self, idx: torch.Tensor, loops: torch.Tensor) -> torch.Tensor:
        """Logits where sample i is read out after its own loop count loops[i] (>= 1)."""
        e = self.wte(idx)
        loops = loops.to(idx.device)
        # Python-side bound for the loop; one host sync per forward is negligible here.
        max_loops = int(loops.max())
        h = torch.zeros_like(e)
        out = torch.zeros_like(e)
        for t in range(1, max_loops + 1):
            h = self.step(h, e)
            # Keep h_t for samples whose K_i == t; gradients flow only through that path.
            out = torch.where((loops == t)[:, None, None], h, out)
        return self.head(out)

    @torch.no_grad()
    def iterate_loops(self, idx: torch.Tensor, max_loops: int):
        """Yield (t, logits after t iterations) for t = 1..max_loops in a single pass."""
        e = self.wte(idx)
        h = torch.zeros_like(e)
        for t in range(1, max_loops + 1):
            h = self.step(h, e)
            yield t, self.head(h)
