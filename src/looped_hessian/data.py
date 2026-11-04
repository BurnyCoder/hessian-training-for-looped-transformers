"""Binary-addition task with length generalization.

Local: generates (input, label) token tensors for a + b in a variant of the
full-output-prediction (FOP) format of Fan et al. 2024 ("Looped Transformers for Length
Generalization", arXiv:2409.15647, Sec. 2.2 and 6.1.1): the input is `a + b >` followed by
n+2 placeholder tokens; the label at each placeholder is one bit of the (n+1)-bit sum,
most significant bit first (Fan et al. explicitly do not reverse the output), then EOS.
Every non-answer position is ignored by the loss. Difference from Fan et al.'s code
(github.com/UW-Madison-Lee-Lab/looped-tf, src/generate_training_data.py): they write the
sum bits starting at the `>` position (a one-position shift) and supervise every trailing
EOS position up to the padded width; here the sum starts at the first placeholder and only
the n+1 sum bits plus one EOS are supervised.

Global: the training loop samples lengths through `curriculum_max_len`, loop counts
through `sample_loop_counts` (the randomized-loop schedule of Kuo et al. 2026,
arXiv:2606.29983), and evaluation uses `make_eval_set` with dedicated seeds so test
problems never share an RNG stream with training.
"""

import torch

# Token vocabulary. EOS doubles as the answer placeholder in the input (Fan et al. fill
# the output locations with EOS tokens); PAD only right-pads mixed-length batches.
ZERO, ONE, PLUS, EOQ, EOS, PAD = range(6)
VOCAB_SIZE = 6
TOKEN_STRINGS = ["0", "1", "+", ">", "#", "_"]
# PyTorch cross_entropy's default ignore_index, used for every unlabelled position.
IGNORE = -100


def _bits_to_int(bits: list[int]) -> int:
    # Interpret a most-significant-bit-first list of 0/1 as a Python int (exact for any length).
    return int("".join(map(str, bits)), 2) if bits else 0


def make_example(a_bits: list[int], b_bits: list[int]) -> tuple[list[int], list[int]]:
    """Build one FOP example from two equal-length MSB-first bit lists."""
    n = len(a_bits)
    # Exact integer sum, rendered with n+1 bits so the carry-out bit is always present.
    total = _bits_to_int(a_bits) + _bits_to_int(b_bits)
    sum_bits = [int(c) for c in format(total, f"0{n + 1}b")]
    # Input: a, '+', b, '>', then n+2 placeholders (n+1 answer bits + 1 EOS slot).
    x = a_bits + [PLUS] + b_bits + [EOQ] + [EOS] * (n + 2)
    # Labels: ignore the question part, supervise the n+1 sum bits and the final EOS.
    y = [IGNORE] * (2 * n + 2) + sum_bits + [EOS]
    return x, y


def make_batch(lengths: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one problem per entry of `lengths` and right-pad to a common length.

    Right padding is safe because the model is causal: padded positions come after every
    real token, so they cannot influence the logits at real positions.
    """
    xs, ys = [], []
    for n in lengths.tolist():
        # Draw the 2n operand bits from the provided generator, so batches are a pure
        # function of the generator state (identical across training arms with one seed).
        bits = torch.randint(0, 2, (2 * n,), generator=gen).tolist()
        x, y = make_example(bits[:n], bits[n:])
        xs.append(x)
        ys.append(y)
    width = max(len(x) for x in xs)
    # Pad inputs with PAD and labels with IGNORE so padding never contributes to the loss.
    x = torch.full((len(xs), width), PAD, dtype=torch.long)
    y = torch.full((len(ys), width), IGNORE, dtype=torch.long)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        x[i, : len(xi)] = torch.tensor(xi)
        y[i, : len(yi)] = torch.tensor(yi)
    return x, y


def curriculum_max_len(step: int, total_steps: int, max_len: int) -> int:
    """Largest training length allowed at `step`.

    Fan et al. start from the smallest length and add one length at fixed intervals; here
    the schedule is linear from 1 at step 0 to `max_len` at the midpoint of training
    (preregistration Sec. 3), so the second half trains on the full length range.
    """
    half = max(1, total_steps // 2)
    return min(max_len, 1 + (step * (max_len - 1)) // half)


def sample_lengths(step: int, total_steps: int, max_len: int, batch_size: int,
                   gen: torch.Generator) -> torch.Tensor:
    # Lengths are uniform on [1, current curriculum maximum].
    n_max = curriculum_max_len(step, total_steps, max_len)
    return torch.randint(1, n_max + 1, (batch_size,), generator=gen)


def sample_loop_counts(lengths: torch.Tensor, gen: torch.Generator, jitter: int,
                       max_loops: int) -> torch.Tensor:
    """Randomized loop counts K_i ~ U{n_i - jitter, ..., n_i + jitter}, clamped to [1, max_loops].

    This is the "randomized K = n +- 5" schedule of Kuo et al. 2026 (arXiv:2606.29983),
    which reported lower across-seed OOD variance than K = n.
    """
    offsets = torch.randint(-jitter, jitter + 1, lengths.shape, generator=gen)
    return (lengths + offsets).clamp(1, max_loops)


def make_eval_set(length: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed test set of `count` problems of one length, from its own seeded generator."""
    gen = torch.Generator().manual_seed(seed)
    return make_batch(torch.full((count,), length), gen)


def decode(tokens: torch.Tensor) -> str:
    # Human-readable rendering used in logs and qualitative output examples.
    return "".join(TOKEN_STRINGS[int(t)] if int(t) >= 0 else "*" for t in tokens)
