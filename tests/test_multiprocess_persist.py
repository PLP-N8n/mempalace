from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from mempalace.backends.chroma import ChromaBackend, evict_shared_system

DIM = 8
COLL = "repro_t"
REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _embedding_for(seed: int) -> list[float]:
    v = [0.0] * DIM
    v[seed % DIM] = 1.0
    v[(seed * 3) % DIM] += 0.5
    return v


def _seed_for_id(id_str: str) -> int:
    return int(hashlib.md5(id_str.encode("utf-8")).hexdigest(), 16) % 1000


def _read_persisted_id_to_label(palace: str, collection_name: str) -> dict[str, int]:
    """Return the ``id_to_label`` map from the persisted HNSW metadata pickle."""
    db_path = os.path.join(palace, "chroma.sqlite3")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = db.execute(
            "select s.id from segments s join collections c on s.collection = c.id "
            "where c.name = ? and s.scope = 'VECTOR'",
            (collection_name,),
        ).fetchone()
        if row is None:
            return {}
        seg_id = row[0]
        pk = Path(palace) / seg_id / "index_metadata.pickle"
        if not pk.exists():
            return {}
        with open(pk, "rb") as f:
            meta = pickle.load(f)
        if isinstance(meta, dict):
            return meta.get("id_to_label", {})
        return getattr(meta, "id_to_label", {})
    finally:
        db.close()


_CHILD_B = r"""
import os, sys
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, {sys_path!r})
from mempalace.backends.chroma import ChromaBackend
palace_path = {palace!r}
ids = {ids!r}
embeddings = {embeddings!r}
backend = ChromaBackend()
col = backend.get_or_create_collection(palace_path, {coll!r})
col.add(
    ids=ids,
    documents=[f"doc {{i}}" for i in ids],
    metadatas=[{{"k": "b"}} for _ in ids],
    embeddings=embeddings,
)
print("child-ok")
"""


_VERIFY_C = r"""
import os, sys, json
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, {sys_path!r})
from mempalace.backends.chroma import ChromaBackend
palace_path = {palace!r}
b_ids = {b_ids!r}
collection_name = {coll!r}
query_embedding = {query_embedding!r}
backend = ChromaBackend()
col = backend.get_or_create_collection(palace_path, collection_name)
out = {{}}
out["count"] = col.count()
try:
    got = col.get(ids=b_ids, include=["embeddings"])
    embs = got.embeddings if hasattr(got, "embeddings") else got.get("embeddings")
    out["b_vectors"] = [bool(e is not None and len(e) > 0) for e in (embs or [])]
except Exception as e:
    out["b_vectors_error"] = repr(e)[:300]
try:
    q = col.query(query_embeddings=[query_embedding], n_results=2, where={{"k": "b"}})
    ids = q.ids if hasattr(q, "ids") else q["ids"]
    out["where_query_ids"] = ids[0]
except Exception as e:
    out["where_query_error"] = repr(e)[:300]
print(json.dumps(out))
"""


def test_multiprocess_persist_keeps_external_writes(tmp_path: Path) -> None:
    """Process A adds rows, process B adds rows, then A writes again via its
    cached backend. B's rows must survive in the persisted HNSW index.
    """
    palace = str(tmp_path)
    a_ids = ["a1", "a2"]
    a3_ids = ["a3", "a4"]
    b_ids = ["b1", "b2"]

    all_ids = a_ids + a3_ids + b_ids
    embeddings = {i: _embedding_for(_seed_for_id(i)) for i in all_ids}

    backend = ChromaBackend()
    col = backend.get_or_create_collection(palace, COLL)
    col.add(
        ids=a_ids,
        documents=[f"doc {i}" for i in a_ids],
        metadatas=[{"k": "a"} for _ in a_ids],
        embeddings=[embeddings[i] for i in a_ids],
    )

    child_code = _CHILD_B.format(
        sys_path=REPO_ROOT,
        palace=palace,
        coll=COLL,
        ids=b_ids,
        embeddings=[embeddings[i] for i in b_ids],
    )
    child = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert child.returncode == 0, child.stderr
    assert "child-ok" in child.stdout, child.stderr

    # A writes again through its cached backend. The fix in ChromaBackend._client
    # must detect the DB change, close and evict chromadb's shared System, and
    # load the on-disk index so B's rows are preserved.
    col2 = backend.get_or_create_collection(palace, COLL)
    col2.add(
        ids=a3_ids,
        documents=[f"doc {i}" for i in a3_ids],
        metadatas=[{"k": "a"} for _ in a3_ids],
        embeddings=[embeddings[i] for i in a3_ids],
    )

    id_to_label = _read_persisted_id_to_label(palace, COLL)
    assert set(b_ids) <= set(id_to_label), (
        f"missing b ids from persisted index: {id_to_label.keys()}"
    )

    verify_code = _VERIFY_C.format(
        sys_path=REPO_ROOT,
        palace=palace,
        coll=COLL,
        b_ids=b_ids,
        query_embedding=embeddings["b1"],
    )
    verify = subprocess.run(
        [sys.executable, "-c", verify_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert verify.returncode == 0, verify.stderr
    result = json.loads(verify.stdout)
    assert result.get("count") == 6, result
    assert result.get("b_vectors") == [True, True], result
    assert result.get("where_query_ids") == b_ids, result

    backend.close()


def test_evict_shared_system_is_narrow(tmp_path: Path) -> None:
    """evict_shared_system removes only the target palace's System key."""
    from chromadb.api.client import SharedSystemClient

    path_a = str(tmp_path / "palace_a")
    path_b = str(tmp_path / "palace_b")

    backend_a = ChromaBackend()
    backend_b = ChromaBackend()
    try:
        backend_a.get_or_create_collection(path_a, COLL)
        backend_b.get_or_create_collection(path_b, COLL)

        keys_before = set(SharedSystemClient._identifier_to_system)
        assert path_a in keys_before, keys_before
        assert path_b in keys_before, keys_before

        evict_shared_system(path_a)

        keys_after = set(SharedSystemClient._identifier_to_system)
        assert path_a not in keys_after, keys_after
        assert path_b in keys_after, keys_after
    finally:
        backend_a.close()
        backend_b.close()
