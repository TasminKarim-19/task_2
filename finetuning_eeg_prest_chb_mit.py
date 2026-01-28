import sys, json, re, warnings
from pathlib import Path
from datetime import datetime
import time, os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import mne
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if not getattr(torch, "_biot_stft_patched", False):
    torch._biot_stft_patched = True
    _orig_stft = torch.stft

    def _stft_with_hann(
        input, n_fft, hop_length=None, win_length=None, window=None,
        center=True, pad_mode="reflect", normalized=False, onesided=None, return_complex=None
    ):
        if window is None:
            wlen = n_fft if win_length is None else win_length
            window = torch.hann_window(wlen, device=input.device, dtype=input.dtype)
        return _orig_stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )

    torch.stft = _stft_with_hann


base_dir = Path(r"C:\Users\user01\PycharmProjects\task")
biot_repo = base_dir / "BIOT"

manifest_dir = base_dir / "chbmit_manifest_paper_10s_stride5_FIXEDPARSE"
manifest_csv = manifest_dir / "manifest.csv"
splits_json = manifest_dir / "splits.json"

train_cache_dir = manifest_dir / "cache_train_npz"
val_cache_npz = manifest_dir / "val_cache_A.npz"
test_cache_npz = manifest_dir / "test_cache_A.npz"

init_ckpt = biot_repo / r"pretrained-models\EEG-PREST-16-channels.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"

target_sfreq = 200
window_sec = 10
n_times = 2000

target_channels_16 = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
]
ch_map = {c: i for i, c in enumerate(target_channels_16)}

token_size = 200
hop_length = 100
n_classes = 2

epochs = 2
freeze_encoder_epochs = 1
max_steps_per_epoch = 200

batch_size_train = 64
batch_size_eval = 256

lr_head_frozen = 3e-4
lr_head = 1e-3
lr_encoder = 3e-5
weight_decay = 1e-2

focal_gamma = 2.0
focal_alpha = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)

num_workers = 0
pin_memory = False

run_name = f"biot_prest16_chbmit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
runs_dir = base_dir / "runs"
log_dir = runs_dir / run_name
runs_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

best_ckpt_path = runs_dir / f"{run_name}_best.pt"
final_json_path = runs_dir / f"{run_name}_final.json"

for p in [biot_repo, manifest_dir, manifest_csv, splits_json, train_cache_dir, init_ckpt]:
    assert p.exists(), f"missing path: {p}"

sys.path.insert(0, str(biot_repo))
from model.biot import BIOTClassifier  # noqa: E402


def norm_ch(name: str) -> str:
    return re.sub(r"-\d+$", "", name.strip())


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def infer_biot_hparams(sd: dict):
    depth = 4
    idxs = []
    for k in sd.keys():
        m = re.search(r"(layers|blocks)\.(\d+)\.", k)
        if m:
            idxs.append(int(m.group(2)))
    if idxs:
        depth = max(idxs) + 1

    emb = 256
    for k, v in sd.items():
        if hasattr(v, "shape") and len(v.shape) == 2 and ("to_q" in k.lower()):
            if v.shape[0] == v.shape[1]:
                emb = int(v.shape[0])
                break

    heads = 16 if emb % 16 == 0 else 8
    return {"emb_size": emb, "depth": depth, "heads": heads}


def build_biot(emb_size=256, depth=4, heads=8):
    return BIOTClassifier(
        emb_size=emb_size,
        heads=heads,
        depth=depth,
        n_classes=n_classes,
        sampling_rate=target_sfreq,
        token_size=token_size,
        hop_length=hop_length,
        sample_length=window_sec,
        n_times=n_times,
        dropout=0.1,
    )


def load_prest16(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    model_sd = model.state_dict()

    filtered = {}
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        if k in model_sd and v.shape == model_sd[k].shape:
            filtered[k] = v
        elif f"biot.{k}" in model_sd and v.shape == model_sd[f"biot.{k}"].shape:
            filtered[f"biot.{k}"] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return model, len(filtered), len(missing), len(unexpected)


class focal_loss(torch.nn.Module):
    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** focal_gamma * ce
        loss = focal_alpha[y] * loss
        return loss.mean()


class cached_train_by_file(Dataset):
    def __init__(self, cache_dir: Path):
        self.files = sorted(list(cache_dir.glob("*.npz")))
        self.lengths = [np.load(f)["y"].shape[0] for f in self.files]
        self.prefix = np.cumsum([0] + self.lengths)
        self.total = int(self.prefix[-1])

        self._cf = None
        self._cx = None
        self._cy = None

    def __len__(self):
        return self.total

    def _load(self, fi):
        f = self.files[fi]
        if self._cf == f:
            return
        d = np.load(f)
        self._cx, self._cy = d["x"], d["y"]
        self._cf = f

    def __getitem__(self, idx):
        fi = int(np.searchsorted(self.prefix, idx, side="right") - 1)
        li = int(idx - self.prefix[fi])
        self._load(fi)
        return torch.tensor(self._cx[li]).float(), torch.tensor(int(self._cy[li])).long()


class cached_npz_dataset(Dataset):
    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.x = torch.from_numpy(d["x"]).float()
        self.y = torch.from_numpy(d["y"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def safe_remove(path: Path, tries=20, wait_s=0.5):
    for _ in range(tries):
        try:
            if path.exists():
                os.remove(str(path))
            return True
        except PermissionError:
            time.sleep(wait_s)
        except Exception:
            time.sleep(wait_s)
    return False


def ensure_eval_cache(df_subset: pd.DataFrame, out_npz: Path):
    if out_npz.exists():
        try:
            d = np.load(out_npz)
            _ = d["x"].shape
            _ = d["y"].shape
            return
        except Exception:
            safe_remove(out_npz)

    tmp = out_npz.with_name(out_npz.name + ".tmp.npz")
    safe_remove(tmp)

    xs = np.zeros((len(df_subset), 16, n_times), dtype=np.float16)
    ys = np.zeros((len(df_subset),), dtype=np.int64)

    last_path = None
    raw = None
    idxs = None

    for i, r in enumerate(df_subset.itertuples(index=False)):
        if r.edf_path != last_path:
            raw = mne.io.read_raw_edf(r.edf_path, preload=False, verbose="ERROR")
            last_path = r.edf_path
            ch_names = [norm_ch(c) for c in raw.ch_names]
            idxs = [None] * 16
            for ci, ch in enumerate(ch_names):
                if ch in ch_map:
                    idxs[ch_map[ch]] = ci

        sf = float(raw.info["sfreq"])
        start = int(round(float(r.start_sec) * sf))
        stop = start + int(round(window_sec * sf))

        x0 = raw.get_data(start=start, stop=stop)
        t_len = stop - start
        x = np.zeros((16, t_len), dtype=np.float32)
        for k in range(16):
            ci = idxs[k]
            if ci is not None:
                x[k] = x0[ci].astype(np.float32)

        xt = torch.tensor(x, dtype=torch.float32)
        xt = xt / (torch.quantile(xt.abs(), 0.95, dim=1, keepdim=True) + 1e-6)
        xt = F.interpolate(xt.unsqueeze(0), size=n_times, mode="linear", align_corners=False).squeeze(0)

        xs[i] = xt.numpy().astype(np.float16)
        ys[i] = int(r.label)

    np.savez_compressed(tmp, x=xs, y=ys)

    for _ in range(20):
        try:
            safe_remove(out_npz)
            os.replace(str(tmp), str(out_npz))
            return
        except PermissionError:
            time.sleep(0.5)

    raise PermissionError(f"could not replace {out_npz} (file locked)")


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    out = {"n_pos": float(n_pos), "n_neg": float(n_neg)}
    out["auc_pr"] = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else float("nan")
    out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    out["bacc"] = float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(np.int32)))
    return out


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    ys, ps = [], []
    total_loss, total_n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        prob = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()

        ys.append(y.cpu().numpy())
        ps.append(prob)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return total_loss / max(1, total_n), compute_metrics(ys, ps)


def freeze_encoder_only(model, freeze: bool):
    for n, p in model.named_parameters():
        if not (p.is_floating_point() or p.is_complex()):
            continue
        if "classifier" in n.lower():
            p.requires_grad = True
        else:
            p.requires_grad = (not freeze)


def opt_head_only(model):
    head = [p for n, p in model.named_parameters() if "classifier" in n.lower() and p.requires_grad]
    return torch.optim.AdamW(head, lr=lr_head_frozen, weight_decay=weight_decay)


def opt_discriminative(model):
    head, enc = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in n.lower():
            head.append(p)
        else:
            enc.append(p)
    return torch.optim.AdamW(
        [{"params": enc, "lr": lr_encoder}, {"params": head, "lr": lr_head}],
        weight_decay=weight_decay,
    )


def main():
    splits = json.loads(splits_json.read_text())["A"]
    df = pd.read_csv(manifest_csv)

    val_df = df[df["patient"].isin(splits["val"])].copy().reset_index(drop=True)
    test_df = df[df["patient"].isin(splits["test"])].copy().reset_index(drop=True)

    ensure_eval_cache(val_df, val_cache_npz)
    ensure_eval_cache(test_df, test_cache_npz)

    ckpt = torch.load(init_ckpt, map_location="cpu")
    sd = extract_state_dict(ckpt)
    inferred = infer_biot_hparams(sd)

    model = build_biot(**inferred).to(device)
    model, loaded, missing, unexpected = load_prest16(model, init_ckpt)

    tr_loader = DataLoader(
        cached_train_by_file(train_cache_dir),
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    va_loader = DataLoader(
        cached_npz_dataset(val_cache_npz),
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    te_loader = DataLoader(
        cached_npz_dataset(test_cache_npz),
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loss_fn = focal_loss()
    writer = SummaryWriter(str(log_dir))

    writer.add_text("run/name", run_name, 0)
    writer.add_text("run/paths", json.dumps({"manifest_dir": str(manifest_dir), "biot_repo": str(biot_repo)}), 0)
    writer.add_text(
        "run/ckpt_load",
        json.dumps({"loaded": loaded, "missing": missing, "unexpected": unexpected, "inferred": inferred}, indent=2),
        0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_aucpr = -1.0
    best_summary = None

    for ep in range(1, epochs + 1):
        if ep <= freeze_encoder_epochs:
            freeze_encoder_only(model, True)
            optimizer = opt_head_only(model)
        else:
            freeze_encoder_only(model, False)
            optimizer = opt_discriminative(model)

        model.train()
        running, seen = 0.0, 0
        global_step = (ep - 1) * max_steps_per_epoch

        for step, (x, y) in enumerate(tr_loader):
            if step >= max_steps_per_epoch:
                break

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            running += float(loss.item()) * bs
            seen += bs

            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

        train_loss = running / max(1, seen)
        val_loss, val_m = evaluate(model, va_loader, loss_fn)
        test_loss, test_m = evaluate(model, te_loader, loss_fn)

        writer.add_scalar("loss/train_epoch", train_loss, ep)
        writer.add_scalar("loss/val", val_loss, ep)
        writer.add_scalar("loss/test", test_loss, ep)
        for k, v in val_m.items():
            writer.add_scalar(f"metrics/val_{k}", v, ep)
        for k, v in test_m.items():
            writer.add_scalar(f"metrics/test_{k}", v, ep)
        writer.flush()

        if not np.isnan(val_m["auc_pr"]) and float(val_m["auc_pr"]) > best_aucpr:
            best_aucpr = float(val_m["auc_pr"])
            torch.save(model.state_dict(), best_ckpt_path)
            best_summary = {
                "epoch": ep,
                "val_loss": float(val_loss),
                "val_metrics": val_m,
                "test_loss": float(test_loss),
                "test_metrics": test_m,
            }

    writer.close()

    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    final_val_loss, final_val_m = evaluate(model, va_loader, loss_fn)
    final_test_loss, final_test_m = evaluate(model, te_loader, loss_fn)

    final_json_path.write_text(
        json.dumps(
            {
                "task": "chb-mit seizure detection (prest-16 finetune)",
                "run_name": run_name,
                "device": device,
                "init_ckpt": str(init_ckpt),
                "inferred_model": inferred,
                "ckpt_load": {"loaded": loaded, "missing": missing, "unexpected": unexpected},
                "best_ckpt": str(best_ckpt_path),
                "log_dir": str(log_dir),
                "best": best_summary,
                "final": {
                    "val_loss": float(final_val_loss),
                    "val_metrics": final_val_m,
                    "test_loss": float(final_test_loss),
                    "test_metrics": final_test_m,
                },
                "config": {
                    "epochs": epochs,
                    "max_steps_per_epoch": max_steps_per_epoch,
                    "freeze_encoder_epochs": freeze_encoder_epochs,
                    "batch_size_train": batch_size_train,
                    "batch_size_eval": batch_size_eval,
                },
            },
            indent=2,
        )
    )

    print(f"val:  loss={final_val_loss:.4f}  auprc={final_val_m['auc_pr']:.4f}  auroc={final_val_m['auroc']:.4f}  bacc={final_val_m['bacc']:.4f}")
    print(f"test: loss={final_test_loss:.4f} auprc={final_test_m['auc_pr']:.4f} auroc={final_test_m['auroc']:.4f} bacc={final_test_m['bacc']:.4f}")


if __name__ == "__main__":
    main()
