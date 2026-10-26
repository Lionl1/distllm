import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "plan_split", Path(__file__).parents[1] / "examples" / "plan_split.py"
)
plan_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_split)

LAYERS = [100_000_000] * 40  # 40 layers of 100M params
EMBED = HEAD = 500_000_000


def _plan(nodes, quantize="nf4", dtype="float16"):
    return plan_split.plan(LAYERS, EMBED, HEAD, kv_width=1024, nodes=nodes,
                           quantize=quantize, dtype=dtype, context=1024)


def test_covers_all_layers_in_order():
    stages = _plan([("a", 10, None), ("b", 10, None), ("c", 10, None)])
    assert stages[0][1] == 0 and stages[-1][2] == 39
    assert all(prev[2] + 1 == nxt[1] for prev, nxt in zip(stages, stages[1:]))


def test_more_memory_gets_more_layers():
    small, big = _plan([("small", 6, None), ("big", 24, None)])
    assert (big[2] - big[1]) > (small[2] - small[1])


def test_fixed_counts_and_shared_machine():
    stages = _plan([("me", 6, 2), ("alice", 10, None), ("bob", 10, None), ("me", 6, 2)])
    assert [s[2] - s[1] + 1 for s in (stages[0], stages[-1])] == [2, 2]
    # "me" holds embeddings + head + 4 layers: both stages fit into its 6 GB together.
    assert stages[0][3] + stages[-1][3] < 6 * plan_split.HEADROOM


def test_refuses_what_does_not_fit():
    with pytest.raises(SystemExit, match="needs"):
        _plan([("a", 2, None), ("b", 2, None)], quantize=None)
