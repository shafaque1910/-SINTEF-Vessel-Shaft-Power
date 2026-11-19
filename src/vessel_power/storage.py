import hashlib
import json
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")


def frame_fingerprint(frame):
    """Hash values, row index, column names and dtypes, independent of pickle bytes."""
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {"columns": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes]},
            sort_keys=True,
        ).encode()
    )
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    return h.hexdigest()
