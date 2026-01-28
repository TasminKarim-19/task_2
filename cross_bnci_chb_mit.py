import os, re, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, f1_score


base_dir = Path(r"C:\Users\user01\PycharmProjects\task")

biot_repo = base_dir / "BIOT"
chb_dir = base_dir / "chbmit_manifest_paper_10s_stride5_FIXEDPARSE"

manifest_csv = chb_dir / "manifest.csv"
splits_json = chb_dir / "splits.json"
train_cache_dir = chb_dir / "cache_train_npz"
val_cache_npz = chb_dir / "val_cache_A.npz"
test_cache_npz = chb_dir / "test_cache_A.npz"

bnci_bundle_path = base_dir / "biot_sleepedf_scratch_best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

target_sfreq = 200
window_sec = 10
n_times = 2000

token_size = 200
hop_length = 100
n_classes = 2

emb_size = 256
depth = 4
heads = 16

epochs = 2
max_steps_per_epoch = 200
freeze_encoder_epochs = 1

batch_size_train = 64
batch_size_eval = 256

lr_head_frozen = 3e-4
lr_head = 1e-3
lr_encoder = 3e-5
weight_decay = 1e-2

focal_gamma = 2.0
focal_alpha = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)

run_name = f"cross_bnci_to_chbmit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
runs_dir = base_dir / "runs"
log_dir = runs_dir / run_name
runs_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

best_ckpt = runs_dir / f"{run_name}_best.pt"
final_json = runs_dir / f"{run_name}_final.json"


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


for p in [biot_repo, chb_dir, manifest_csv, splits_json, train_cache_dir, val_cache_npz, test_cache_npz, bnci_bundle_path]:
    assert p.exists(), f"missing path: {p}"


import sys
sys.path.insert(0, str(biot_repo))
from model.biot import BIOTClassifier  # noqa: E402


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


class focal_loss(torch.nn.Module):
    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** focal_gamma * ce
        loss = focal_alpha[y] * loss
        return loss.mean()


def build_biot():
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


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    out = {"n_pos": float(n_pos), "n_neg": float(n_neg)}
    out["auc_pr"] = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else float("nan")
    out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    out["bacc"] = float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(np.int32)))
    out["f1"] = float(f1_score(y_true, (y_prob >= 0.5).astype(np.int32), zero_division=0))
    return out


@torch.no_grad()
def eval_loss_and_metrics(model, loader, loss_fn):
    model.eval()
    ys, ps = [], []
    total_loss, total_n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        ys.append(y.detach().cpu().numpy())
        ps.append(prob)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return total_loss / max(1, total_n), compute_metrics(ys, ps)


@torch.no_grad()
def eval_probs(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    return np.concatenate(ys), np.concatenate(ps)


def freeze_encoder_only(model, freeze: bool):
    for n, p in model.named_parameters():
        if not (p.is_floating_point() or p.is_complex()):
            continue
        if "classifier" in n.lower():
            p.requires_grad = True
        else:
            p.requires_grad = (not freeze)


def opt_head_only(model):
    head = [
        p for n, p in model.named_parameters()
        if (p.is_floating_point() or p.is_complex()) and "classifier" in n.lower() and p.requires_grad
    ]
    return torch.optim.AdamW(head, lr=lr_head_frozen, weight_decay=weight_decay)


def opt_discriminative(model):
    head, enc = [], []
    for n, p in model.named_parameters():
        if not (p.is_floating_point() or p.is_complex()):
            continue
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


def load_cross_encoder(model, bundle_path: Path):
    bundle = torch.load(bundle_path, map_location="cpu")
    src = bundle["state_dict"] if isinstance(bundle, dict) and "state_dict" in bundle else bundle
    tgt = model.state_dict()

    loaded = {}
    for k, v in src.items():
        if "classifier" in k.lower():
            continue
        if k in tgt and hasattr(v, "shape") and v.shape == tgt[k].shape:
            loaded[k] = v

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    return model, len(loaded), len(missing), len(unexpected)


def qk_to_attention(q_out, k_out, n_heads, head_dim):
    B, N, E = q_out.shape
    q = q_out.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)
    k = k_out.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    return torch.softmax(scores, dim=-1)


def plot_attn(attn, title):
    a = attn.mean(dim=1)[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(a, cmap="hot")
    ax.set_title(title)
    ax.set_xlabel("token")
    ax.set_ylabel("token")
    fig.colorbar(im, ax=ax)
    return fig


def log_attention(writer, model, val_cache_path: Path, val_loader, step=0):
    yv, _ = eval_probs(model, val_loader)
    pos = np.where(yv == 1)[0]
    idx0 = int(pos[0]) if len(pos) else 0

    d = np.load(val_cache_path)
    x0 = torch.from_numpy(d["x"][idx0]).float().unsqueeze(0).to(device)

    q_store, k_store = [], []
    hooks = []

    def hook_q(m, i, o):
        q_store.append(o.detach())

    def hook_k(m, i, o):
        k_store.append(o.detach())

    for name, mod in model.named_modules():
        ln = name.lower()
        if ln.endswith("to_q") and isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m, i, o: hook_q(m, i, o)))
        if ln.endswith("to_k") and isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m, i, o: hook_k(m, i, o)))

    with torch.no_grad():
        _ = model(x0)

    for h in hooks:
        h.remove()

    if not (len(q_store) and len(k_store)):
        writer.add_text("attention/status", "could not hook to_q/to_k", step)
        writer.flush()
        return

    n_blocks = min(len(q_store), len(k_store))
    attn_list = [
        qk_to_attention(q_store[i], k_store[i], heads, emb_size // heads).cpu()
        for i in range(n_blocks)
    ]

    early = attn_list[0]
    mid = attn_list[len(attn_list) // 2]
    late = attn_list[-1]

    writer.add_figure("attention/early", plot_attn(early, "early attention (qk)"), step)
    writer.add_figure("attention/mid", plot_attn(mid, "mid attention (qk)"), step)
    writer.add_figure("attention/late", plot_attn(late, "late attention (qk)"), step)
    writer.flush()


def main():
    splits = json.loads(splits_json.read_text())["A"]
    df = pd.read_csv(manifest_csv)

    val_df = df[df["patient"].isin(splits["val"])].copy()
    test_df = df[df["patient"].isin(splits["test"])].copy()

    val_pos = int((val_df["label"] == 1).sum())
    test_pos = int((test_df["label"] == 1).sum())

    model = build_biot().to(device)
    model, n_loaded, n_missing, n_unexp = load_cross_encoder(model, bnci_bundle_path)

    loss_fn = focal_loss()

    tr_loader = DataLoader(cached_train_by_file(train_cache_dir), batch_size=batch_size_train, shuffle=True)
    va_loader = DataLoader(cached_npz_dataset(val_cache_npz), batch_size=batch_size_eval, shuffle=False)
    te_loader = DataLoader(cached_npz_dataset(test_cache_npz), batch_size=batch_size_eval, shuffle=False)

    writer = SummaryWriter(str(log_dir))
    writer.add_text(
        "run/info",
        json.dumps(
            {
                "run_name": run_name,
                "device": device,
                "bnci_bundle": str(bnci_bundle_path),
                "cross_loaded": n_loaded,
                "missing": n_missing,
                "unexpected": n_unexp,
                "val_pos": val_pos,
                "test_pos": test_pos,
            },
            indent=2,
        ),
        0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_aucpr = -1.0
    best_summary = None
    best_epoch = None

    for ep in range(1, epochs + 1):
        if ep <= freeze_encoder_epochs:
            freeze_encoder_only(model, True)
            optimizer = opt_head_only(model)
        else:
            freeze_encoder_only(model, False)
            optimizer = opt_discriminative(model)

        model.train()
        run_loss, seen = 0.0, 0
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
            run_loss += float(loss.item()) * bs
            seen += bs

            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

        train_loss = run_loss / max(1, seen)
        val_loss, val_m = eval_loss_and_metrics(model, va_loader, loss_fn)
        test_loss, test_m = eval_loss_and_metrics(model, te_loader, loss_fn)

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
            best_epoch = ep
            torch.save(model.state_dict(), best_ckpt)
            best_summary = {"epoch": ep, "val_loss": float(val_loss), "val": val_m, "test_loss": float(test_loss), "test": test_m}

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    final_val_loss, final_val_m = eval_loss_and_metrics(model, va_loader, loss_fn)
    final_test_loss, final_test_m = eval_loss_and_metrics(model, te_loader, loss_fn)

    log_attention(writer, model, val_cache_npz, va_loader, step=0)
    writer.close()

    final_json.write_text(
        json.dumps(
            {
                "run": run_name,
                "paths": {
                    "base_dir": str(base_dir),
                    "bnci_bundle": str(bnci_bundle_path),
                    "chb_dir": str(chb_dir),
                    "best_ckpt": str(best_ckpt),
                    "log_dir": str(log_dir),
                },
                "cross_transfer": {"loaded": n_loaded, "missing": n_missing, "unexpected": n_unexp},
                "best": {"epoch": best_epoch, "val_auc_pr": best_aucpr, "summary": best_summary},
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
                    "heads": heads,
                    "emb_size": emb_size,
                    "depth": depth,
                },
            },
            indent=2,
        )
    )

    print(
        f"val : loss={final_val_loss:.4f} auprc={final_val_m['auc_pr']:.4f} "
        f"auroc={final_val_m['auroc']:.4f} bacc={final_val_m['bacc']:.4f} f1={final_val_m['f1']:.4f}"
    )
    print(
        f"test: loss={final_test_loss:.4f} auprc={final_test_m['auc_pr']:.4f} "
        f"auroc={final_test_m['auroc']:.4f} bacc={final_test_m['bacc']:.4f} f1={final_test_m['f1']:.4f}"
    )
    print(rf"tensorboard --logdir {runs_dir}")


if __name__ == "__main__":
    main()
