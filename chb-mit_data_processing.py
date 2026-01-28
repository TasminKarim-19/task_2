import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

chb_root = Path(r"C:\Users\user01\PycharmProjects\task\chb-mit-scalp-eeg-database-1.0.0")
out_dir = Path(r"C:\Users\user01\PycharmProjects\task\chbmit_manifest_paper_10s_stride5_FIXEDPARSE")
out_dir.mkdir(parents=True, exist_ok=True)

window_sec = 10
stride_sec = 5

seiz_start_re = re.compile(r"Seizure\s*(?:\d+\s*)?Start Time:\s*(\d+)\s*seconds", re.IGNORECASE)
seiz_end_re = re.compile(r"Seizure\s*(?:\d+\s*)?End Time:\s*(\d+)\s*seconds", re.IGNORECASE)
file_re = re.compile(r"File Name:\s*(.+\.edf)", re.IGNORECASE)
num_re = re.compile(r"\d+(?:\.\d+)?", re.IGNORECASE)


def patient_id_from_name(name: str) -> int:
    m = re.match(r"chb(\d+)", name.lower())
    return int(m.group(1)) if m else -1


def parse_summary(summary_path: Path):
    lines = summary_path.read_text(errors="ignore").splitlines()
    file_seiz = {}
    cur_file = None
    cur_start = None

    for line in lines:
        s = line.strip()

        m = file_re.match(s)
        if m:
            cur_file = m.group(1).strip()
            file_seiz.setdefault(cur_file, [])
            cur_start = None
            continue

        m = seiz_start_re.match(s)
        if m and cur_file is not None:
            cur_start = float(m.group(1))
            continue

        m = seiz_end_re.match(s)
        if m and cur_file is not None and cur_start is not None:
            end = float(m.group(1))
            file_seiz[cur_file].append((cur_start, end))
            cur_start = None

    return file_seiz


def parse_seizures_file(seiz_path: Path):
    try:
        lines = seiz_path.read_text(errors="ignore").splitlines()
    except Exception:
        return []

    intervals = []
    pending = None

    for line in lines:
        s = line.strip().lower()
        nums = num_re.findall(s)

        if "start" in s and nums:
            pending = float(nums[0])
            if "end" in s and len(nums) >= 2:
                a, b = float(nums[0]), float(nums[1])
                if b > a:
                    intervals.append((a, b))
                pending = None
            continue

        if "end" in s and pending is not None and nums:
            b = float(nums[0])
            if b > pending:
                intervals.append((pending, b))
            pending = None
            continue

        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            if b > a:
                intervals.append((a, b))
                pending = None

    intervals = [(max(0.0, a), max(0.0, b)) for a, b in intervals if b > a]
    return intervals


def overlaps(t0, t1, intervals):
    for a, b in intervals:
        if not (t1 <= a or t0 >= b):
            return True
    return False


def main():
    train_p = [f"chb{i:02d}" for i in range(1, 20)]
    val_p = [f"chb{i:02d}" for i in range(20, 22)]
    test_p = [f"chb{i:02d}" for i in range(22, 24)]

    split_a = {"train": train_p, "val": val_p, "test": test_p}
    split_b = {"train": train_p, "val": test_p, "test": val_p}
    (out_dir / "splits.json").write_text(json.dumps({"A": split_a, "B": split_b}, indent=2))

    pat_dirs = sorted(
        [p for p in chb_root.glob("chb*") if p.is_dir()],
        key=lambda p: patient_id_from_name(p.name),
    )

    rows = []
    for p_dir in pat_dirs:
        patient = p_dir.name
        pid = patient_id_from_name(patient)
        if pid < 1 or pid > 23:
            continue

        summary_path = p_dir / f"{patient}-summary.txt"
        summary_map = parse_summary(summary_path) if summary_path.exists() else {}

        for edf_path in sorted(p_dir.glob("*.edf")):
            edf_name = edf_path.name

            seiz_path = p_dir / f"{edf_name}.seizures"
            intervals = parse_seizures_file(seiz_path) if seiz_path.exists() else []
            if not intervals:
                intervals = summary_map.get(edf_name, [])

            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            dur_sec = raw.n_times / sfreq

            starts = np.arange(0, dur_sec - window_sec, stride_sec, dtype=np.float32)
            labels = np.array(
                [1 if overlaps(s, s + window_sec, intervals) else 0 for s in starts],
                dtype=np.int8,
            )

            for i, s in enumerate(starts):
                rows.append(
                    {
                        "patient": patient,
                        "patient_id": pid,
                        "edf_path": str(edf_path),
                        "sfreq": sfreq,
                        "start_sec": float(s),
                        "end_sec": float(s + window_sec),
                        "label": int(labels[i]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest.csv", index=False)

    (out_dir / "settings.json").write_text(
        json.dumps(
            {
                "window_sec": window_sec,
                "stride_sec": stride_sec,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
