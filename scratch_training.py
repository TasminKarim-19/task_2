import sys, json, warnings, re
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", message="A window was not provided.*spectral leakage.*", category=UserWarning)


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
data_dir = base_dir / "biot_bnci2014_001_prepared_200hz_4s"

split_path = data_dir / "split_subjects.json"
meta_path = data_dir / "meta.json"

run_root = base_dir / "runs"
run_root.mkdir(parents=True, exist_ok=True)

assert biot_repo.exists(), rf"BIOT repo not found at: {biot_repo}"
assert data_dir.exists(), rf"prepared BNCI folder not found at: {data_dir}"

sys.path.insert(0, str(biot_repo))
from model.biot import BIOTClassifier  # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

sfreq = 200
n_classes = 2

token_size = 200
hop_length = 100

batch_size = 64
epochs = 60
lr = 3e-4
num_workers = 2

eval_test_every = 5
attn_log_every = 5

use_zscore = False

run_name = f"biot_bnci_scratch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = run_root / run_name
log_dir.mkdir(parents=True, exist_ok=True)

best_state_path = run_root / f"{run_name}_best_state_dict.pt"
bundle_path = run_root / f"{run_name}_bundle.pt"
final_json_path = run_root / f"{run_name}_final.json"


def load_meta(meta_file: Path):
    rows = json.loads(meta_file.read_text())
    return {m["subject"]: {"npz": m["npz"], "n_epochs": int(m["n_epochs"])} for m in rows}


class bnci_npz_dataset(Dataset):
    def __init__(self, subjects, meta_dict, normalize=False):
        self.subjects = list(subjects)
        self.meta = meta_dict
        self.normalize = normalize

        self.lengths = [self.meta[s]["n_epochs"] for s in self.subjects]
        self.prefix = np.cumsum([0] + self.lengths)
        self.total = int(self.prefix[-1])

        self._subj = None
        self._x = None
        self._y = None

    def __len__(self):
        return self.total

    def _load(self, subj):
        if self._subj == subj:
            return
        with np.load(self.meta[subj]["npz"]) as d:
            self._x = d["x"]
            self._y = d["y"]
        self._subj = subj

    def __getitem__(self, idx):
        si = int(np.searchsorted(self.prefix, idx, side="right") - 1)
        subj = self.subjects[si]
        ei = int(idx - self.prefix[si])
        self._load(subj)

        x = torch.from_numpy(self._x[ei]).float()
        y = torch.tensor(int(self._y[ei]), dtype=torch.long)

        if self.normalize:
            m = x.mean(dim=-1, keepdim=True)
            s = x.std(dim=-1, keepdim=True) + 1e-6
            x = (x - m) / s

        return x, y


class subject_batch_sampler(Sampler):
    def __init__(self, prefix, batch_size, seed=42):
        self.prefix = prefix
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        ranges = [(int(self.prefix[i]), int(self.prefix[i + 1])) for i in range(len(self.prefix) - 1)]
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(ranges)
        for a, b in ranges:
            for i in range(a, b, self.batch_size):
                yield list(range(i, min(i + self.batch_size, b)))

    def __len__(self):
        total = int(self.prefix[-1])
        return (total + self.batch_size - 1) // self.batch_size


def metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total = 0.0
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total += float(loss.item()) * x.size(0)
        preds.append(torch.argmax(logits, 1).cpu().numpy())
        labels.append(y.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return float(total / len(loader.dataset)), metrics(y_true, y_pred)


def class_weights(train_subjects, meta_dict):
    counts = np.zeros(n_classes, dtype=np.int64)
    for s in train_subjects:
        with np.load(meta_dict[s]["npz"]) as d:
            y = d["y"]
        for c in range(n_classes):
            counts[c] += int((y == c).sum())

    w = counts.sum() / (counts + 1e-9)
    w = w / w.mean()
    return counts, torch.tensor(w, dtype=torch.float32, device=device)


def build_model(n_times, epoch_sec, n_channels):
    return BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=n_classes,
        sampling_rate=sfreq,
        token_size=token_size,
        hop_length=hop_length,
        sample_length=epoch_sec,
        n_times=n_times,
        n_channels=n_channels,
        n_fft=token_size,
        dropout=0.1,
    )


def get_layers(transformer: nn.Module):
    if hasattr(transformer, "layers") and isinstance(getattr(transformer, "layers"), nn.ModuleList):
        return list(transformer.layers)

    pat = re.compile(r"(layers|blocks)\.(\d+)\b")
    hits = []
    for name, m in transformer.named_modules():
        mm = pat.search(name)
        if mm:
            hits.append((int(mm.group(2)), m))

    if not hits:
        return []

    by = {}
    for idx, m in hits:
        if idx not in by:
            by[idx] = m
    return [by[i] for i in sorted(by.keys())]


def affinity(tokens, max_tokens=160):
    if not torch.is_tensor(tokens) or tokens.ndim != 3:
        return None

    if tokens.shape[1] > tokens.shape[2]:
        z = tokens[0].transpose(0, 1)
    else:
        z = tokens[0]

    L = min(z.shape[0], max_tokens)
    z = z[:L].float()
    z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
    a = (z @ z.T).clamp(-1, 1)
    return a.detach().cpu().numpy()


def log_attention(writer, model, loader, epoch, tag="attention"):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)

    enc = getattr(model, "biot", None)
    if enc is None:
        writer.add_text(f"{tag}/status", "no model.biot found", epoch)
        return

    tr = getattr(enc, "transformer", None)
    if tr is None:
        writer.add_text(f"{tag}/status", "no encoder.transformer found", epoch)
        return

    layers = get_layers(tr)
    captured = {}
    hooks = []

    def hook_fn(key):
        def _h(module, inp, out):
            if torch.is_tensor(out):
                captured[key] = out.detach()
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if torch.is_tensor(o):
                        captured[key] = o.detach()
                        break
        return _h

    if len(layers) >= 3:
        picks = [0, len(layers) // 2, len(layers) - 1]
        keys = ["early", "mid", "late"]
        for k, idx in zip(keys, picks):
            hooks.append(layers[idx].register_forward_hook(hook_fn(k)))
    else:
        hooks.append(tr.register_forward_hook(hook_fn("late")))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if not captured:
        writer.add_text(f"{tag}/status", "no transformer outputs captured", epoch)
        return

    for k in ["early", "mid", "late"]:
        if k not in captured:
            continue
        a = affinity(captured[k], max_tokens=160)
        if a is None:
            continue
        fig = plt.figure(figsize=(5, 4))
        plt.imshow(a, aspect="auto")
        plt.colorbar()
        plt.title(f"{tag}: {k} layer (token affinity)")
        plt.xlabel("token")
        plt.ylabel("token")
        writer.add_figure(f"{tag}/{k}", fig, global_step=epoch)
        plt.close(fig)


def main():
    assert meta_path.exists(), f"meta.json not found: {meta_path}"
    assert split_path.exists(), f"split_subjects.json not found: {split_path}"

    meta = load_meta(meta_path)
    split = json.loads(split_path.read_text())

    tr_ds = bnci_npz_dataset(split["train"], meta, normalize=use_zscore)
    va_ds = bnci_npz_dataset(split["val"], meta, normalize=use_zscore)
    te_ds = bnci_npz_dataset(split["test"], meta, normalize=use_zscore)

    tr_sampler = subject_batch_sampler(tr_ds.prefix, batch_size=batch_size, seed=seed)

    pin = (device == "cuda")
    tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, num_workers=num_workers, pin_memory=pin)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    xb, _ = next(iter(tr_loader))
    n_channels = int(xb.shape[1])
    n_times = int(xb.shape[2])
    epoch_sec = int(round(n_times / sfreq))
    print("detected:", "channels=", n_channels, "n_times=", n_times, "epoch_sec~", epoch_sec)

    model = build_model(n_times=n_times, epoch_sec=epoch_sec, n_channels=n_channels).to(device)
    emb_n = int(model.biot.channel_tokens.num_embeddings)
    print("model channel token capacity:", emb_n)
    assert emb_n >= n_channels, f"model n_channels={emb_n} < data channels={n_channels}"

    counts, w = class_weights(split["train"], meta)
    print("train class counts:", counts.tolist(), "weights:", w.detach().cpu().numpy().round(3).tolist())

    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config/run_name", run_name, 0)
    writer.add_text("config/split", json.dumps(split), 0)
    writer.add_text(
        "config/model",
        json.dumps(
            {
                "emb_size": 256,
                "heads": 8,
                "depth": 4,
                "n_classes": n_classes,
                "sfreq": sfreq,
                "token_size": token_size,
                "hop_length": hop_length,
                "n_channels": n_channels,
                "n_times": n_times,
                "epoch_sec": epoch_sec,
                "use_zscore_training": use_zscore,
            },
            indent=2,
        ),
        0,
    )
    print("tensorboard log dir:", log_dir)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_f1 = -1.0
    step = 0

    for epoch in range(1, epochs + 1):
        tr_sampler.set_epoch(epoch)
        model.train()

        run_loss = 0.0
        seen = 0

        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            run_loss += float(loss.item()) * bs
            seen += bs

            writer.add_scalar("loss/train_step", float(loss.item()), step)
            step += 1

        tr_loss = float(run_loss / max(1, seen))
        va_loss, va_m = evaluate(model, va_loader, crit)

        writer.add_scalar("loss/train_epoch", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        for k, v in va_m.items():
            writer.add_scalar(f"metrics/val_{k}", v, epoch)

        if epoch % eval_test_every == 0 or epoch == epochs:
            te_loss, te_m = evaluate(model, te_loader, crit)
            writer.add_scalar("loss/test", te_loss, epoch)
            for k, v in te_m.items():
                writer.add_scalar(f"metrics/test_{k}", v, epoch)

        if epoch % attn_log_every == 0 or epoch == epochs:
            log_attention(writer, model, va_loader, epoch, tag="attention")

        writer.flush()

        print(
            f"epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
            f"| val_acc={va_m['acc']:.4f} | val_f1_macro={va_m['f1_macro']:.4f} | val_kappa={va_m['kappa']:.4f}"
        )

        if va_m["f1_macro"] > best_f1:
            best_f1 = va_m["f1_macro"]
            torch.save(model.state_dict(), best_state_path)
            print("  saved best:", best_state_path)

    writer.close()

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    model.to(device)
    te_loss, te_m = evaluate(model, te_loader, crit)

    bundle = {
        "state_dict": model.state_dict(),
        "config": {
            "emb_size": 256,
            "heads": 8,
            "depth": 4,
            "n_classes": n_classes,
            "sampling_rate": sfreq,
            "token_size": token_size,
            "hop_length": hop_length,
            "sample_length": epoch_sec,
            "n_times": n_times,
            "n_channels": n_channels,
            "n_fft": token_size,
            "dropout": 0.1,
            "seed": seed,
            "dataset": "bnci2014_001 left_right_imagery",
            "split": split,
            "use_zscore_training": use_zscore,
        },
    }
    torch.save(bundle, bundle_path)

    final_json_path.write_text(
        json.dumps(
            {
                "task": "bnci2014_001_left_right_mi",
                "model": "biotclassifier (official repo) - scratch",
                "final_test_loss": float(te_loss),
                "final_test_metrics": te_m,
                "log_dir": str(log_dir),
                "best_state_dict": str(best_state_path),
                "bundle_path": str(bundle_path),
                "bundle_config": bundle["config"],
            },
            indent=2,
        )
    )

    print("\nfinal (best checkpoint)")
    print("test_loss:", te_loss)
    print("test_metrics:", te_m)
    print("bundle:", bundle_path)
    print("json:", final_json_path)
    print("\ntensorboard:")
    print(rf"  tensorboard --logdir {run_root}")


if __name__ == "__main__":
    main()
