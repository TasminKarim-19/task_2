import json
import numpy as np
from pathlib import Path

from moabb import set_download_dir, setup_seed
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

out_dir = Path(r"C:\Users\user01\PycharmProjects\task\biot_bnci2014_001_prepared_200hz_4s")
out_dir.mkdir(parents=True, exist_ok=True)

cache_dir = Path(r"C:\Users\user01\PycharmProjects\task\moabb_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

seed = 42
sfreq = 200
tmin, tmax = 0.0, 4.0
fmin, fmax = 8.0, 35.0

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15


def norm95(x_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x_ct.astype(np.float32, copy=False)
    s = np.percentile(np.abs(x), 95, axis=1, keepdims=True).astype(np.float32)
    s = np.maximum(s, eps)
    x = x / s
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def make_split(subjects, y_by_subj, seed: int = 42, max_tries: int = 300):
    rng = np.random.default_rng(seed)
    subs = list(subjects)

    def has_both(lbls: np.ndarray) -> bool:
        return (0 in lbls) and (1 in lbls)

    for _ in range(max_tries):
        rng.shuffle(subs)
        n = len(subs)

        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * val_ratio)))
        n_te = n - n_tr - n_va
        if n_te < 1:
            n_te = 1
            n_va = max(1, n - n_tr - n_te)

        tr = subs[:n_tr]
        va = subs[n_tr:n_tr + n_va]
        te = subs[n_tr + n_va:]

        y_tr = np.concatenate([y_by_subj[s] for s in tr])
        y_va = np.concatenate([y_by_subj[s] for s in va])
        y_te = np.concatenate([y_by_subj[s] for s in te])

        if has_both(y_tr) and has_both(y_va) and has_both(y_te):
            return {"train": tr, "val": va, "test": te}

    raise RuntimeError("could not create a balanced train/val/test split; try another seed")


def map_labels(y):
    y = np.asarray(y)
    if y.dtype.kind in ("U", "S", "O"):
        y_str = np.array([str(v) for v in y], dtype=object)
        m = {
            "left_hand": 0, "left": 0, "0": 0,
            "right_hand": 1, "right": 1, "1": 1,
        }
        unk = sorted(set(y_str) - set(m.keys()))
        if unk:
            raise ValueError(f"unexpected labels: {unk}")
        return np.array([m[v] for v in y_str], dtype=np.int64)
    return y.astype(np.int64)


def main():
    setup_seed(seed)
    set_download_dir(str(cache_dir))

    ds = BNCI2014_001()
    pdm = LeftRightImagery(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=sfreq)

    subs = ds.subject_list
    print("subjects:", subs)

    x, y, meta = pdm.get_data(dataset=ds, subjects=subs)
    x = x.astype(np.float32)
    y = map_labels(y)

    print("loaded:", x.shape, "labels:", np.bincount(y))

    subj_arr = meta["subject"].to_numpy()
    uniq = sorted(np.unique(subj_arr).tolist())

    meta_rows = []
    y_by_subj = {}

    for s in uniq:
        idx = np.where(subj_arr == s)[0]
        xs = x[idx]
        ys = y[idx]

        xs2 = np.empty_like(xs, dtype=np.float32)
        for i in range(xs.shape[0]):
            xs2[i] = norm95(xs[i])

        f = out_dir / f"subj_{int(s):02d}.npz"
        np.savez_compressed(f, x=xs2, y=ys)

        meta_rows.append({
            "subject": int(s),
            "npz": str(f),
            "n_epochs": int(len(ys)),
            "n_channels": int(xs.shape[1]),
            "n_times": int(xs.shape[2]),
        })
        y_by_subj[int(s)] = ys

    (out_dir / "meta.json").write_text(json.dumps(meta_rows, indent=2))
    split = make_split([r["subject"] for r in meta_rows], y_by_subj, seed=seed)
    (out_dir / "split_subjects.json").write_text(json.dumps(split, indent=2))

    settings = {
        "dataset": "BNCI2014_001",
        "paradigm": "LeftRightImagery",
        "seed": seed,
        "sfreq": sfreq,
        "tmin": tmin,
        "tmax": tmax,
        "fmin": fmin,
        "fmax": fmax,
        "normalization": "per-trial per-channel abs 95th percentile",
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    print("prepared subjects:", len(meta_rows))
    print("split:", split)
    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
