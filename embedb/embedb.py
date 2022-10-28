import io
import os.path
from typing import List, Sequence, Union

import lmdb
import numpy as np


def bytes_to_vector(x: bytes) -> np.ndarray:
    return np.frombuffer(x, dtype=np.float32)


def vector_to_bytes(x: np.ndarray) -> bytes:
    return x.tobytes()


def ndarray_to_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return buffer.read()


def bytes_to_ndarray(b: bytes) -> np.ndarray:
    buffer = io.BytesIO(b)
    buffer.seek(0)
    return np.load(buffer)


class EmbeDB:
    """
    A simple file-based key-value store for embeddings. Powered by LMDB.
    """

    def __init__(
        self,
        path: str,
        readonly=True,
        size=None,
        float_vector_only=True,
        encoder=None,
        decoder=None,
    ):
        """
        :param path: Path to the database.
        :param readonly: Whether the database is read-only.
        :param size: Maximum size of the database in bytes.
        :param float_vector_only: When True, only 1D float32 array are allowed (speed optimized).
        :param encoder: custom encoder function with a signature (obj => bytes).
        :param decoder: custom decoder function with a signature (bytes => obj).
        """
        size = size or 10**12  # approx 1 TB
        self.path = path

        db_dir = os.path.dirname(path)
        os.makedirs(db_dir, exist_ok=True)

        self.env = lmdb.Environment(
            path=path, map_size=size, lock=False, readonly=readonly, subdir=False
        )
        self.encoder = (
            encoder or vector_to_bytes if float_vector_only else ndarray_to_bytes
        )
        self.decoder = (
            decoder or bytes_to_vector if float_vector_only else bytes_to_ndarray
        )

    def __getitem__(
        self, item: Union[Sequence[str], str]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        with self.env.begin(write=False) as txn:
            if isinstance(item, str):
                res = txn.get(item.encode())
                return self.decoder(res) if res else None

            if isinstance(item, (list, tuple, np.ndarray)):
                cur = txn.cursor()
                res = [
                    self.decoder(v) if v else None
                    for _, v in cur.getmulti([x.encode() for x in item])
                ]
                cur.close()
                return res

        raise ValueError(f"Invalid type for item: {type(item)} {item}")

    def __setitem__(self, key: str, value: np.ndarray):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), self.encoder(value))

    def __contains__(self, item):
        return self[item] is not None

    def get(self, key: str) -> np.ndarray:
        return self[key]

    def keys(self):
        with self.env.begin() as txn:
            cur = txn.cursor()
            for k, _ in cur:
                yield k.decode()
            cur.close()

    def values(self):
        with self.env.begin() as txn:
            cur = txn.cursor()
            for _, v in cur:
                yield self.decoder(v)
            cur.close()

    def items(self):
        with self.env.begin() as txn:
            cur = txn.cursor()
            for k, v in cur:
                yield k.decode(), self.decoder(v)
            cur.close()

    def update(self, other):
        with self.env.begin(write=True) as txn:
            cur = txn.cursor()
            pairs = [(k.encode(), self.encoder(v)) for k, v in other.items()]
            cur.putmulti(pairs)
            cur.close()

    def __len__(self):
        with self.env.begin() as txn:
            length = txn.stat()["entries"]
        return length

    def close(self):
        self.env.close()

    def __del__(self):
        self.close()

    def delete(self, item: Union[Sequence[str], str]):
        with self.env.begin(write=True) as txn:
            if isinstance(item, str):
                return txn.delete(item.encode())

            if isinstance(item, (list, tuple, np.ndarray)):
                return [txn.delete(x.encode()) for x in item]
