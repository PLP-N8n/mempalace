"""A dim-None segment that has had DELETIONS must not be quarantined.

`total_elements_added` is monotonic: it counts every add ever made to the
segment. `id_to_label` / `label_to_id` hold only the elements still live. So
any segment that has had a single deletion carries
``total_elements_added > len(id_to_label)`` — the ordinary, healthy shape of a
used index.

Requiring equality therefore condemns every post-deletion dim-None segment as
unrecoverable, and `quarantine_invalid_hnsw_metadata` renames it to
`.corrupt-<stamp>`. Upstream fixed this in #1710 by requiring `>=` rather than
`==`; this fork carried the `!=` form until 2026-09-02.

Observed, not hypothetical. On DESKTOP-8CLL4OC on 2026-09-01 this quarantined
two live indexes whose metadata was structurally perfect and whose label maps
were bijective:

    segment bf5271f0   total_elements_added=3515   live labels=3514   -> 1 deletion
    segment 0214e611   total_elements_added=2959   live labels=2917   -> 42 deletions

**One deleted drawer destroyed a 3,514-vector index.** These tests use those
exact numbers so the regression is named by the incident that produced it.
"""

import pickle
from pathlib import Path

from mempalace.backends.chroma import (
    _HNSW_MISSING_METADATA_DATA_FLOOR,
    _missing_dimensionality_appears_recoverable,
)


def _segment(tmp_path: Path, *, total: int, live: int) -> tuple[dict, dict, str]:
    """A dim-None segment with `live` elements and `total` adds over its lifetime."""
    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "data_level0.bin").write_bytes(b"\0" * (_HNSW_MISSING_METADATA_DATA_FLOOR + 1))
    (seg_dir / "link_lists.bin").write_bytes(b"\0" * 128)

    id_to_label = {f"id-{i}": i for i in range(live)}
    label_to_id = {label: item_id for item_id, label in id_to_label.items()}
    persisted = {
        "dimensionality": None,
        "total_elements_added": total,
        "max_seq_id": None,
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
    }
    (seg_dir / "index_metadata.pickle").write_bytes(pickle.dumps(persisted, protocol=3))
    return persisted, id_to_label, str(seg_dir)


def test_one_deletion_does_not_make_a_segment_unrecoverable(tmp_path):
    """The exact shape of the 2026-09-01 incident: 3515 adds, 3514 live."""
    persisted, id_to_label, seg_dir = _segment(tmp_path, total=3515, live=3514)
    assert _missing_dimensionality_appears_recoverable(persisted, id_to_label, seg_dir) is True


def test_many_deletions_do_not_make_a_segment_unrecoverable(tmp_path):
    persisted, id_to_label, seg_dir = _segment(tmp_path, total=2959, live=2917)
    assert _missing_dimensionality_appears_recoverable(persisted, id_to_label, seg_dir) is True


def test_a_segment_with_no_deletions_is_still_recoverable(tmp_path):
    """The equality case must keep working — the fix widens, it does not move."""
    persisted, id_to_label, seg_dir = _segment(tmp_path, total=3514, live=3514)
    assert _missing_dimensionality_appears_recoverable(persisted, id_to_label, seg_dir) is True


def test_fewer_adds_than_live_labels_is_still_rejected(tmp_path):
    """The guard must keep its teeth. total < live is impossible for a sane
    segment — adds are monotonic and deletions only ever remove labels — so it
    means the metadata really is inconsistent. This is the case `>=` still
    catches, and the reason the fix is `<` and not "drop the check"."""
    persisted, id_to_label, seg_dir = _segment(tmp_path, total=10, live=3514)
    assert _missing_dimensionality_appears_recoverable(persisted, id_to_label, seg_dir) is False


def test_a_broken_label_map_is_still_rejected(tmp_path):
    """Bijection check survives the widening."""
    persisted, id_to_label, seg_dir = _segment(tmp_path, total=3515, live=3514)
    persisted["label_to_id"] = {label: "wrong" for label in id_to_label.values()}
    assert _missing_dimensionality_appears_recoverable(persisted, id_to_label, seg_dir) is False


def test_a_mismatched_label_map_size_is_still_rejected(tmp_path):
    persisted, id_to_label, seg_dir = _segment(tmp_path, total=3515, live=3514)
    persisted["label_to_id"] = {}
    assert _missing_dimensionality_appears_recoverable(persisted, id_to_label, seg_dir) is False
