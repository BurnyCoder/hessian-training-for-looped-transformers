"""Tests for the binary-addition task (format of Fan et al. 2024, arXiv:2409.15647, Sec. 6.1.1).

Global context: every reported accuracy depends on these generators producing correct
labels, so the tests decode generated tensors back to integers and check a + b.
"""

import torch

from looped_hessian import data


def _decode_bits(tokens):
    # Map token ids back to a binary string (only ZERO/ONE are expected here).
    return "".join({data.ZERO: "0", data.ONE: "1"}[int(t)] for t in tokens)


def test_example_format_and_sum_are_correct():
    gen = torch.Generator().manual_seed(0)
    for n in [1, 2, 7, 19, 60]:
        x, y = data.make_batch(torch.full((32,), n), gen)
        # Sequence length is n (a) + 1 (+) + n (b) + 1 (>) + n+2 (answer placeholders).
        assert x.shape == (32, 3 * n + 4) and y.shape == x.shape
        for row_x, row_y in zip(x, y):
            a = _decode_bits(row_x[:n])
            b = _decode_bits(row_x[n + 1 : 2 * n + 1])
            assert row_x[n] == data.PLUS and row_x[2 * n + 1] == data.EOQ
            assert torch.all(row_x[2 * n + 2 :] == data.EOS)
            # Only the n+2 answer positions carry labels (full-output prediction, no shift).
            assert torch.all(row_y[: 2 * n + 2] == data.IGNORE)
            answer = _decode_bits(row_y[2 * n + 2 : 3 * n + 3])
            assert row_y[-1] == data.EOS
            # The answer is the (n+1)-bit sum, most significant bit first, leading zeros kept.
            assert int(answer, 2) == int(a, 2) + int(b, 2) and len(answer) == n + 1


def test_mixed_lengths_are_right_padded_with_ignored_labels():
    gen = torch.Generator().manual_seed(1)
    lengths = torch.tensor([1, 5, 3])
    x, y = data.make_batch(lengths, gen)
    assert x.shape == (3, 3 * 5 + 4)
    for i, n in enumerate(lengths.tolist()):
        assert torch.all(x[i, 3 * n + 4 :] == data.PAD)
        assert torch.all(y[i, 3 * n + 4 :] == data.IGNORE)
        assert int((y[i] != data.IGNORE).sum()) == n + 2


def test_generation_is_deterministic_given_seed():
    a = data.make_batch(torch.tensor([4, 9]), torch.Generator().manual_seed(7))
    b = data.make_batch(torch.tensor([4, 9]), torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_curriculum_is_linear_from_1_to_max_at_half_training():
    assert data.curriculum_max_len(0, 1000, 19) == 1
    assert data.curriculum_max_len(500, 1000, 19) == 19
    assert data.curriculum_max_len(999, 1000, 19) == 19
    values = [data.curriculum_max_len(s, 1000, 19) for s in range(1000)]
    assert values == sorted(values)


def test_loop_counts_are_jittered_around_length_and_clamped():
    gen = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 20, (5000,), generator=gen)
    loops = data.sample_loop_counts(lengths, gen, jitter=5, max_loops=20)
    assert loops.min() >= 1 and loops.max() <= 20
    assert torch.all((loops - lengths).abs() <= 5)
    # Unclamped region should hit both extremes of the jitter window.
    mid = lengths == 10
    assert set(loops[mid].tolist()) == set(range(5, 16))


def test_eval_sets_use_independent_seeds():
    x1, _ = data.make_eval_set(20, 50, seed=1)
    x2, _ = data.make_eval_set(20, 50, seed=2)
    assert not torch.equal(x1, x2)
