# EmbeDB
EmbeDB is a small Python wrapper around [LMDB](https://lmdb.readthedocs.io/) built as key-value storage for embeddings.

Installation: 
`pip install git+https://github.com/ntropy-network/embedb`

## Usage

```python
from embedb import EmbeDB
import numpy as np

size = 1000000
vectors = np.random.rand(size, 512).astype('float32')
keys = [str(i) for i in range(size)]

d = {k: v for k, v in zip(keys, vectors)}
db = EmbeDB("/tmp/test.db", readonly=False)
db.update(d)  # faster version of iterations on db[k] = v

subset = np.random.choice(keys, 10000)
subset_vectors = db[subset]  # faster version of [d[k] for k in subset]
```

Basic benchmark:

```python 
In [2]: !ls -lha /tmp/test.db
-rw-r--r--  1 Arseny  wheel   2.8G Nov  4 18:39 /tmp/test.db

In [3]: %timeit _ = db[subset]
21.2 ms ± 598 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [4]: %timeit _ = [d[k] for k in subset]
2.57 ms ± 36 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
