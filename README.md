# EmbeDB
EmbeDB is a small Python wrapper around LMDB built as key-value storage for embeddings.

Installation: 
`pip install git+https://github.com/ntropy-network/embedb`

## Usage

```python
from embedb import EmbeDB
import numpy as np

vectors = np.random.rand(10000, 512, dtype="float32")
keys = [str(i) for i in range(10000)]

d = {k: v for k, v in zip(keys, vectors)}
db = EmbeDB("/tmp/test.db", readonly=False)
db.update(d)  # faster version of iterations on db[k] = v

subset = np.random.choice(keys, 1000)
subset_vectors = db[subset]  # faster version of [d[k] for k in subset]
```
