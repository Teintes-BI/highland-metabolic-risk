import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp_run_id():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def file_sha256(path: Path, chunk_size=1024 * 1024):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
