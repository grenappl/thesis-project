"""Run once from WSL2 before scripts/train_vggish_ridge.py:

    uv run python scripts/build_panns_pca.py

Compress the Kaggle PANNs embeddings (491k x 2048 float32 = 4 GB, which
has OOM'd WSL2 before) to 256 dims with IncrementalPCA, streaming the
compressed .npz member so peak memory stays well under 1 GB:

  pass 1: partial_fit on every 3rd 4096-row chunk (~164k rows)
  pass 2: transform every row -> 491k x 256 float32 (~500 MB)

Writes data/kaggle_embeddings/panns_pca256.npz (id, features) and the
fitted transform to models/panns/panns_pca256.joblib — the same transform
has to be applied to the live PANNs embedding at inference time.
"""

import io
import zipfile

import joblib
import numpy as np
from numpy.lib import format as npformat
from sklearn.decomposition import IncrementalPCA

NPZ = "data/kaggle_embeddings/panns_embeddings.npz"
COMPONENTS = 256
CHUNK = 4096


def stream_chunks():
    z = zipfile.ZipFile(NPZ)
    with z.open("features.npy") as f:
        version = npformat.read_magic(f)
        reader = npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0
        (n, dim), _, dtype = reader(f)
        row_bytes = dim * np.dtype(dtype).itemsize
        for start in range(0, n, CHUNK):
            count = min(CHUNK, n - start)
            yield start, np.frombuffer(f.read(count * row_bytes), dtype=dtype).reshape(count, dim)


ipca = IncrementalPCA(n_components=COMPONENTS)
for k, (start, block) in enumerate(stream_chunks()):
    if k % 3 == 0:
        ipca.partial_fit(block)
    if k % 20 == 0:
        print(f"pass 1: row {start}", flush=True)
print(f"explained variance ({COMPONENTS} comps): {ipca.explained_variance_ratio_.sum():.3f}", flush=True)

out = []
for k, (start, block) in enumerate(stream_chunks()):
    out.append(ipca.transform(block).astype(np.float32))
    if k % 20 == 0:
        print(f"pass 2: row {start}", flush=True)

ids = np.load(io.BytesIO(zipfile.ZipFile(NPZ).read("id.npy")), allow_pickle=True)
np.savez("data/kaggle_embeddings/panns_pca256.npz", id=ids, features=np.concatenate(out))
joblib.dump(ipca, "models/panns/panns_pca256.joblib")
print("saved panns_pca256.npz + panns_pca256.joblib", flush=True)
