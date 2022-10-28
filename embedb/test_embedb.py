import os
import shutil
import tempfile

import lmdb
import numpy as np
import pytest

from embedb import EmbeDB


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp()
    fname = os.path.join(d, "test.rdb")
    db = EmbeDB(path=fname, readonly=False)

    db["a"] = np.array([1, 2, 3]).astype(np.float32)
    db["b"] = np.array([4, 5, 6]).astype(np.float32)
    assert len(db) == 2
    db.close()

    yield fname

    shutil.rmtree(d)


def test_embedb_getitem_contains(db_path):
    db = EmbeDB(db_path)

    assert "a" in db
    assert "b" in db
    assert db["a"].tolist() == [1, 2, 3]
    assert db["b"].tolist() == [4, 5, 6]
    db.close()


def test_embedb_setitem(db_path):
    db = EmbeDB(db_path, readonly=False)
    db["c"] = np.array([7, 8, 9]).astype(np.float32)
    assert db["c"].tolist() == [7, 8, 9]
    db.close()


def test_embedb_batch_getitem(db_path):
    db = EmbeDB(db_path)

    a, b = db[["a", "b"]]
    assert a.tolist() == [1, 2, 3]
    assert b.tolist() == [4, 5, 6]

    a, b = db[np.array(["a", "b"])]
    assert a.tolist() == [1, 2, 3]
    assert b.tolist() == [4, 5, 6]
    db.close()


def test_embedb_update(db_path):
    db = EmbeDB(db_path, readonly=False)
    pairs = {
        "a": np.array([7, 8, 9]).astype(np.float32),
        "b": np.array([10, 11, 12]).astype(np.float32),
    }
    db.update(pairs)
    db.close()

    db = EmbeDB(db_path)
    assert db["a"].tolist() == [7, 8, 9]
    assert db["b"].tolist() == [10, 11, 12]


def test_embedb_iter(db_path):
    db = EmbeDB(db_path)
    for k, v in db.items():
        assert k in ["a", "b"]
        assert v.tolist() in [[1, 2, 3], [4, 5, 6]]


def test_embedb_keys(db_path):
    db = EmbeDB(db_path)
    assert set(db.keys()) == {"a", "b"}


def test_embedb_values(db_path):
    db = EmbeDB(db_path)
    assert [x.tolist() for x in db.values()] == [[1, 2, 3], [4, 5, 6]]


def test_multidim_embedb(db_path):
    db = EmbeDB(db_path, readonly=False, float_vector_only=False)
    db["c"] = np.array([[1, 2], [3, 4]])
    assert db["c"].tolist() == [[1, 2], [3, 4]]
    db.close()


def test_db_close(db_path):
    db = EmbeDB(db_path)
    db.close()
    with pytest.raises(lmdb.Error):
        _ = db["a"]


def test_db_contains(db_path):
    db = EmbeDB(db_path)
    assert "a" in db
    assert "b" in db
    assert "c" not in db
    assert db["c"] is None
    assert db.get("c") is None
    db.close()


def test_db_delete_single(db_path):
    db = EmbeDB(db_path, readonly=False)
    db.delete("a")
    assert "a" not in db
    assert "b" in db
    db.close()


def test_db_delete_multiple(db_path):
    db = EmbeDB(db_path, readonly=False)
    db.delete(["a", "b"])
    assert "a" not in db
    assert "b" not in db
    db.close()


def test_db_delete_non_existing(db_path):
    db = EmbeDB(db_path, readonly=False)
    deleted = db.delete(["a", "b", "c"])
    assert deleted == [True, True, False]
    db.close()
