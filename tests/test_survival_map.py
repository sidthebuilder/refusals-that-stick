"""End-to-end test of the survival map on small, real GGUF files.

Writes four GGUF files with gguf-py, the same library llama.cpp ships, in
memory-sized form:

* before/after at full precision (F32), standing in for the BF16 conversions;
* before/after quantized to Q8_0 with gguf-py's reference quantizer.

The map must then (1) skip a tensor the repair did not touch, (2) find the
edited one, and (3) report survival consistent with ``compare_edits`` run on the
same arrays directly. Also checks C-ΔΘ's parameter-share rule.
"""

from __future__ import annotations

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

from refusals_that_stick.circuit import ctheta_matched_share  # noqa: E402
from refusals_that_stick.survival import compare_edits  # noqa: E402
from refusals_that_stick.survival_map import module_of, survival_map  # noqa: E402

ROWS, COLS = 8, 64  # COLS is a multiple of the Q8_0 block size (32)


def _write(path, tensors: dict[str, np.ndarray], qtype: GGMLQuantizationType | None):
    w = GGUFWriter(str(path), arch="llama")
    for name, arr in tensors.items():
        if qtype is None:
            w.add_tensor(name, arr.astype(np.float32))
        else:
            # quantize() returns uint8 rows of packed blocks; the writer derives the
            # element shape from that byte shape and the quant type.
            q = quantize(arr.astype(np.float32), qtype)
            w.add_tensor(name, q, raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture()
def pair(tmp_path):
    rng = np.random.default_rng(0)
    before = {
        "blk.0.attn_q.weight": rng.normal(0, 0.02, (ROWS, COLS)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.normal(0, 0.02, (ROWS, COLS)).astype(np.float32),
    }
    after = {k: v.copy() for k, v in before.items()}
    # Edit only the attention tensor, by a large amount (several Q8_0 bins).
    after["blk.0.attn_q.weight"] += 0.01
    paths = {}
    for tag, tensors, qt in (("db", before, None), ("rb", after, None),
                             ("dq", before, GGMLQuantizationType.Q8_0),
                             ("rq", after, GGMLQuantizationType.Q8_0)):
        paths[tag] = tmp_path / f"{tag}.gguf"
        _write(paths[tag], tensors, qt)
    return before, after, paths


def test_module_of():
    assert module_of("blk.3.attn_q.weight") == "attn_q"
    assert module_of("blk.12.ffn_down.weight") == "ffn_down"
    assert module_of("token_embd.weight") == "token_embd"
    assert module_of("output_norm.weight") == "output_norm"


def test_map_skips_untouched_and_measures_edited(pair):
    before, after, p = pair
    out = survival_map(str(p["db"]), str(p["rb"]), str(p["dq"]), str(p["rq"]), qtype_label="Q8_0")
    rows = {r["name"]: r for r in out["tensors"]}

    assert rows["blk.0.ffn_down.weight"]["n_edited"] == 0
    assert out["n_tensors_edited"] == 1
    edited = rows["blk.0.attn_q.weight"]
    assert edited["n_edited"] == ROWS * COLS
    # A shift of several bins survives Q8_0 almost everywhere, in the right direction.
    assert edited["directional_permanence"] > 0.95
    assert out["directional_permanence"] == pytest.approx(edited["directional_permanence"])
    assert out["share_of_weights_edited"] == pytest.approx(0.5)


def test_map_matches_compare_edits_directly(pair):
    before, after, p = pair
    out = survival_map(str(p["db"]), str(p["rb"]), str(p["dq"]), str(p["rq"]), qtype_label="Q8_0")
    name = "blk.0.attn_q.weight"
    qt = GGMLQuantizationType.Q8_0
    deq = lambda a: dequantize(quantize(a, qt), qt).reshape(a.shape)  # noqa: E731
    expected = compare_edits(after[name].astype(np.float64) - before[name],
                             deq(after[name]).astype(np.float64) - deq(before[name]))
    got = {r["name"]: r for r in out["tensors"]}[name]
    assert got["permanence"] == pytest.approx(expected.permanence)
    assert got["directional_permanence"] == pytest.approx(expected.directional_permanence)


def test_map_records_bin_width(pair):
    _, _, p = pair
    out = survival_map(str(p["db"]), str(p["rb"]), str(p["dq"]), str(p["rq"]), qtype_label="Q8_0")
    edited = [r for r in out["tensors"] if r.get("n_edited")][0]
    # Q8_0 step for N(0, 0.02) weights is about max|w|/127 per block: well under 0.01.
    assert 0 < edited["bin_width_est"] < 0.01


# --- C-ΔΘ parameter-share rule ----------------------------------------------------------

SHAPES = {
    "model.embed_tokens.weight": [100, 10],
    "model.layers.0.self_attn.q_proj.weight": [10, 10],
    "model.layers.0.mlp.down_proj.weight": [10, 10],
    "model.layers.1.self_attn.q_proj.weight": [10, 10],
    "model.layers.1.mlp.down_proj.weight": [10, 10],
    "model.layers.1.input_layernorm.weight": [10],
    "model.norm.weight": [10],
}
TOTAL = 1000 + 400 + 10 + 10


def test_share_all_weights_when_no_filter():
    share, n = ctheta_matched_share(SHAPES, None, None)
    assert n == 7
    assert share == pytest.approx(1.0)


def test_share_layer_and_module_filter():
    share, n = ctheta_matched_share(SHAPES, [1], ["q_proj"])
    assert n == 1
    assert share == pytest.approx(100 / TOTAL)


def test_share_layer_only_includes_norms_in_that_layer():
    # SafeTune matches every ".weight" key whose first integer is the layer.
    share, n = ctheta_matched_share(SHAPES, [1], None)
    assert n == 3
    assert share == pytest.approx(210 / TOTAL)


def test_share_keys_without_layer_index_are_skipped_under_layer_filter():
    share, n = ctheta_matched_share(SHAPES, [0, 1], ["proj"])
    assert n == 4  # embed_tokens and model.norm carry no layer index
    assert share == pytest.approx(400 / TOTAL)
