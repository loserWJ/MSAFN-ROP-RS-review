"""Train manuscript MSAFN directly from three fixed dataset files.

python train.py --dataset-dir reviewer_package_10853/data --output runs/msafn
No split generation or row-level partitioning occurs during training.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler

from m import MSAFN, CompositeRobustLoss
from fixed_dataset import DATASET_DIR, SPLIT_NAMES, load_fixed_datasets


def load_legacy_data(path_or_data):
    data = (torch.load(path_or_data, map_location="cpu", weights_only=True)
            if isinstance(path_or_data, (str, Path)) else path_or_data)
    required = ("construction", "surrounding rock", "mwd", "out")
    if not all(k in data for k in required):
        raise ValueError("Data must contain construction, surrounding rock, mwd, out.")
    n = len(data["out"])
    shapes = {"construction": (n, 6, 30), "surrounding rock": (n, 5),
              "mwd": (n, 7, 75, 9), "out": (n, 9)}
    for key, shape in shapes.items():
        if tuple(data[key].shape) != shape or not torch.isfinite(data[key]).all():
            raise ValueError(f"{key}: expected finite tensor of shape {shape}.")
    # Verified against dataset_creat/sets.py and data_shown.py:
    # depth, ROP, hammer, feed, rotation pressure, water pressure, flow, RS, MSE.
    # Current MSE is derived from current ROP/RS, so it must not be an input.
    return (
        data["mwd"][..., 1:].float().contiguous(),
        data["construction"].transpose(1, 2).float().contiguous(),
        data["surrounding rock"].float(),
        data["out"][:, [2, 3, 4, 5, 6]].float(),
        data["out"][:, [1, 7]].float(),
    )


def fit_scalers(tensors, train_indices):
    """Fit each modality featurewise on training rows ONLY."""
    scalers = []
    for x in tensors:
        train = x[train_indices]
        axes = tuple(range(train.ndim - 1))
        low, high = train.amin(dim=axes), train.amax(dim=axes)
        scale = high - low
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        scalers.append({"min": low, "scale": scale})
    return scalers


def transform(tensors, scalers):
    # Do not clamp validation/test values: that would hide extrapolation.
    return tuple((x - s["min"]) / s["scale"] for x, s in zip(tensors, scalers))


class CompleteBatchSampler(Sampler):
    """Keep all training rows, merging a singleton tail for BatchNorm."""

    def __init__(self, size, batch_size, seed):
        if size < 2 or batch_size < 2:
            raise ValueError("Training and batch_size must contain at least 2 samples.")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        batches = list(torch.randperm(self.size, generator=self.generator).split(self.batch_size))
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = torch.cat((batches[-2], batches[-1]))
            batches.pop()
        yield from (batch.tolist() for batch in batches)

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(count > 1 and self.size % self.batch_size == 1)


def epoch_loss(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    sums = torch.zeros(2, device=device)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            *inputs, target = [item.to(device) for item in batch]
            prediction = torch.cat(model(*inputs), dim=1)
            task_losses = loss_fn.per_task(prediction, target)
            tasks = task_losses * prediction.new_tensor(loss_fn.task_weights)
            if not torch.isfinite(tasks).all():
                raise FloatingPointError("Nonfinite loss; check inputs, scaling and optimizer.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                tasks.sum().backward()
                optimizer.step()
            sums += task_losses.detach() * len(target)
            count += len(target)
    if not count:
        raise ValueError("Empty data loader.")
    return (sums / count).cpu().tolist()


@torch.no_grad()
def evaluate(model, loader, target_scaler, device):
    model.eval()
    truths, predictions = [], []
    for batch in loader:
        *inputs, target = batch
        predictions.append(torch.cat(model(*(x.to(device) for x in inputs)), dim=1).cpu())
        truths.append(target)
    y = torch.cat(truths) * target_scaler["scale"] + target_scaler["min"]
    p = torch.cat(predictions) * target_scaler["scale"] + target_scaler["min"]
    metrics = {}
    for col, name in enumerate(("ROP", "RS")):
        error = p[:, col] - y[:, col]
        denom = ((y[:, col] - y[:, col].mean()) ** 2).sum()
        nonzero = y[:, col] != 0
        metrics[name] = {
            "R2": float(1 - error.square().sum() / denom) if denom > 0 else None,
            "MAE": float(error.abs().mean()),
            "MSE": float(error.square().mean()),
            "MAPE_percent": float((error[nonzero] / y[nonzero, col]).abs().mean() * 100)
            if nonzero.any() else None,
            "MAPE_excluded_zero_targets": int((~nonzero).sum()),
        }
    return torch.cat((y, p), dim=1), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=Path("runs/msafn"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--current-input-mode", choices=("separate", "joint10"), default="separate")
    parser.add_argument("--quantile-mode", choices=("formula", "overprediction"), default="formula")
    parser.add_argument("--task-weights", type=float, nargs=2, metavar=("ROP", "RS"), default=(1.0, 1.0),
                        help="Nonnegative task weights; normalized to a weighted mean, default 1:1.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error("Require epochs >= 1, batch-size >= 2, lr > 0.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    raw_datasets, manifest = load_fixed_datasets(args.dataset_dir)
    raw_datasets = {name: load_legacy_data(raw_datasets[name]) for name in SPLIT_NAMES}
    scalers = fit_scalers(raw_datasets["train"],
                          torch.arange(len(raw_datasets["train"][-1])))
    loaders = {}
    for name in SPLIT_NAMES:
        dataset = TensorDataset(*transform(raw_datasets[name], scalers))
        if name == "train":
            sampler = CompleteBatchSampler(len(dataset), args.batch_size, args.seed)
            loaders[name] = DataLoader(dataset, batch_sampler=sampler)
        else:
            loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = MSAFN(current_input_mode=args.current_input_mode).to(args.device)
    loss_fn = CompositeRobustLoss(quantile_mode=args.quantile_mode, task_weights=args.task_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.85, patience=10, min_lr=1e-7)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Fixed dataset counts:", {k: manifest["splits"][k]["rows"] for k in SPLIT_NAMES})
    print("Composite weights are manuscript-reported; GWO is not rerun.")
    print("Task weights (normalized ROP,RS):", loss_fn.task_weights)
    print("Quantile mode:", args.quantile_mode, "; current inputs:", args.current_input_mode)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        train_loss = epoch_loss(model, loaders["train"], loss_fn, args.device, optimizer)
        valid_loss = epoch_loss(model, loaders["validation"], loss_fn, args.device)
        score = sum(w * value for w, value in zip(loss_fn.task_weights, valid_loss))
        scheduler.step(score)
        history.append({"epoch": epoch, "train_ROP": train_loss[0], "train_RS": train_loss[1],
                        "val_ROP": valid_loss[0], "val_RS": valid_loss[1],
                        "val_total": score, "lr": optimizer.param_groups[0]["lr"]})
        if score < best:
            best = score
            torch.save({
                "model_state": model.state_dict(), "model_config": model.config,
                "scalers": scalers,
                "dataset_sha256": {name: manifest["splits"][name]["sha256"]
                                   for name in SPLIT_NAMES},
                "epoch": epoch, "validation_loss": score,
                "loss_weights": loss_fn.weights, "task_weights": loss_fn.task_weights,
                "quantile_mode": args.quantile_mode,
                "data_columns": {"active": [2, 3, 4, 5, 6], "targets": [1, 7]},
                "seed": args.seed, "torch_version": str(torch.__version__),
            }, args.output / "best.pt")
        print(f"Epoch {epoch:03d} train={sum(train_loss)/2:.6f} validation={score:.6f}")
    checkpoint = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    # Test data is evaluated only after selecting the best validation checkpoint.
    values, metrics = evaluate(model, loaders["test"], scalers[-1], args.device)
    with open(args.output / "test_predictions.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_ROP", "true_RS", "pred_ROP", "pred_RS"])
        writer.writerows(values.tolist())
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
