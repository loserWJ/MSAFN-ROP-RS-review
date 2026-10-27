"""Train selected CapacityMSAFN once from fixed files, then evaluate held-out test.

Use only after choosing the three loss weights on validation data. This script
never re-splits observations and computes test metrics only after training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from fixed_dataset import DATASET_DIR, load_fixed_datasets
from gwo_composite_search import fit_candidate, load_data
from m_capacity import CapacityMSAFN
from train import evaluate, load_legacy_data, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=float, nargs=3,
                        metavar=("HUBER", "QUANTILE", "MAE"))
    source.add_argument("--gwo-result", type=Path,
                        help="result.json from a completed GWO search")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be >=1 and batch-size >=2")
    if args.gwo_result:
        gwo = json.loads(args.gwo_result.read_text(encoding="utf-8"))
        if not gwo.get("completed") or gwo.get("profile") == "smoke":
            parser.error("A completed non-smoke GWO result is required")
        weights = tuple(gwo["best"]["weights"])
    else:
        weights = tuple(args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    data = load_data(args.dataset_dir)
    if args.gwo_result:
        config_path = args.gwo_result.with_name("config.json")
        if not config_path.is_file():
            parser.error("GWO config.json is required beside result.json")
        gwo_config = json.loads(config_path.read_text(encoding="utf-8"))
        if (gwo_config.get("profile") != "paper" or
                gwo_config.get("manifest_sha256") != data["manifest_sha256"] or
                gwo_config.get("split_sha256") != data["split_sha256"]):
            parser.error("GWO result does not match the fixed dataset and paper profile")
    fitness, mape, model_state = fit_candidate(
        weights, data, args.epochs, args.seed, args.batch_size, args.device)
    model = CapacityMSAFN(**data["protocol"]["architecture"]).to(args.device)
    model.load_state_dict(model_state)
    # The test file is loaded here, after all training and weight selection.
    test_raw, manifest = load_fixed_datasets(args.dataset_dir, names=("test",))
    test_tensors = transform(load_legacy_data(test_raw["test"]),
                             data["checkpoint"]["scalers"])
    test_loader = DataLoader(TensorDataset(*test_tensors),
                             batch_size=args.batch_size, shuffle=False)
    values, metrics = evaluate(model, test_loader,
                               data["checkpoint"]["scalers"][-1], args.device)
    torch.save({
        "model_class": "CapacityMSAFN", "model_config": data["protocol"]["architecture"],
        "model_state": model_state, "scalers": data["checkpoint"]["scalers"],
        "loss_weights": weights, "task_weights": (.9, .1),
        "quantile_mode": "formula", "seed": args.seed,
        "epochs": args.epochs, "dataset_split_sha256": data["split_sha256"],
    }, args.output / "model.pt")
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("true_ROP", "true_RS", "pred_ROP", "pred_RS"))
        writer.writerows(values.tolist())
    result = {
        "loss_weights": weights, "task_weights_ROP_RS": (.9, .1),
        "training_epochs": args.epochs,
        "validation_MAPE_ROP_RS": mape,
        "validation_weighted_MAPE": fitness,
        "test_metrics": metrics,
        "test_rows": manifest["splits"]["test"]["rows"],
        "dataset_split_sha256": data["split_sha256"],
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
