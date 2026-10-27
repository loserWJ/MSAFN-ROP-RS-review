"""Grey Wolf Optimization outside MSAFN training for three loss coefficients.

Paper profile: 30 wolves, 100 iterations, 100 fixed training epochs per wolf.
The smoke profile only checks the complete implementation and is not a paper
optimization result. No test rows or labels are loaded into candidate fitness.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fixed_dataset import DATASET_DIR, load_fixed_datasets, sha256
from m import CompositeRobustLoss
from m_capacity import CapacityMSAFN
from train import CompleteBatchSampler, fit_scalers, load_legacy_data, transform


PROFILES = {"paper": (30, 100, 100), "smoke": (4, 2, 2)}
TASK_WEIGHTS = (0.9, 0.1)


def save_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def repair_triangle(position) -> np.ndarray:
    """Repair (lambda_H, lambda_Q) into [0,1]^2 with sum <= 1."""
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("A wolf position must contain two finite values")
    point = np.clip(point, 0.0, 1.0)
    if point.sum() > 1.0:
        point = point / point.sum()
    return point


def coefficients(position) -> tuple[float, float, float]:
    point = repair_triangle(position)
    # Subtraction defines lambda_M. Keep the exact float sum accepted by the
    # loss constructor; this is not an independently optimized third axis.
    weights = (float(point[0]), float(point[1]), float(max(0.0, 1.0 - point.sum())))
    return weights


def move_wolves(population: np.ndarray, leaders: list[dict], a: float,
               rng: np.random.Generator) -> np.ndarray:
    """Original alpha/beta/delta encircling update followed by feasible repair."""
    if len(leaders) < 3:
        raise ValueError("GWO needs alpha, beta and delta wolves")
    moved = np.empty_like(population)
    for index, wolf in enumerate(population):
        proposals = []
        for leader in leaders[:3]:
            leader_position = np.asarray(leader["position"], dtype=np.float64)
            r1, r2 = rng.random(2), rng.random(2)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D = np.abs(C * leader_position - wolf)
            proposals.append(leader_position - A * D)
        moved[index] = repair_triangle(np.mean(proposals, axis=0))
    return moved


def load_data(folder: Path):
    # Deliberately load only training and validation files. The test file is
    # untouched during all GWO candidate evaluations.
    raw_datasets, manifest = load_fixed_datasets(folder, names=("train", "validation"))
    train_raw = load_legacy_data(raw_datasets["train"])
    validation_raw = load_legacy_data(raw_datasets["validation"])
    scalers = fit_scalers(train_raw, torch.arange(len(train_raw[-1])))
    train_tensors = transform(train_raw, scalers)
    validation_tensors = transform(validation_raw, scalers)
    return {
        "protocol": {"architecture": manifest["architecture"],
                     "selection_bias": manifest["selection_bias_from_preprocessing"]},
        "checkpoint": {"scalers": scalers},
        "dataset_sha256": manifest["source_dataset_sha256"],
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "split_sha256": {name: manifest["splits"][name]["sha256"]
                         for name in ("train", "validation", "test")},
        "train_tensors": train_tensors,
        "validation_tensors": validation_tensors,
    }


def validation_mape(model, loader, target_scaler, device: str) -> tuple[float, float]:
    model.eval()
    errors = torch.zeros(2, dtype=torch.float64)
    counts = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            *inputs, target = [value.to(device) for value in batch]
            prediction = torch.cat(model(*inputs), dim=1)
            truth = target * target_scaler["scale"].to(device) + target_scaler["min"].to(device)
            output = prediction * target_scaler["scale"].to(device) + target_scaler["min"].to(device)
            for task in range(2):
                nonzero = truth[:, task] != 0
                errors[task] += ((output[nonzero, task] - truth[nonzero, task]) /
                                 truth[nonzero, task]).abs().double().sum().cpu()
                counts[task] += int(nonzero.sum())
    if (counts == 0).any():
        raise ValueError("MAPE undefined because a task has only zero validation targets")
    return tuple((100.0 * errors / counts).tolist())


def fit_candidate(weights: tuple[float, float, float], data, epochs: int,
                  seed: int, batch_size: int, device: str):
    """Fresh model and fixed training order for each wolf; no test access."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    model = CapacityMSAFN(**data["protocol"]["architecture"]).to(device)
    criterion = CompositeRobustLoss(weights=weights, delta=1.5, tau=.75,
                                    quantile_mode="formula", task_weights=TASK_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    training = TensorDataset(*data["train_tensors"])
    validation = TensorDataset(*data["validation_tensors"])
    train_loader = DataLoader(training,
                              batch_sampler=CompleteBatchSampler(len(training), batch_size, seed))
    val_loader = DataLoader(validation, batch_size=batch_size)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            *inputs, target = [value.to(device) for value in batch]
            prediction = torch.cat(model(*inputs), dim=1)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite candidate loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    mape_rop, mape_rs = validation_mape(model, val_loader,
                                        data["checkpoint"]["scalers"][-1], device)
    fitness = TASK_WEIGHTS[0] * mape_rop + TASK_WEIGHTS[1] * mape_rs
    if not math.isfinite(fitness):
        raise FloatingPointError("Nonfinite GWO fitness")
    model_state = {name: value.detach().cpu().clone()
                   for name, value in model.state_dict().items()}
    del model, optimizer
    return fitness, (mape_rop, mape_rs), model_state


def initialize_state(rng, population_size: int):
    initial = rng.dirichlet(np.ones(3), size=population_size)[:, :2]
    return {
        "iteration": 0, "next_wolf": 0,
        "population": initial.tolist(), "leaders": [], "evaluations": [],
        "rng_state": rng.bit_generator.state,
    }


def update_leaders(leaders, record):
    leaders.append({"fitness": record["fitness"], "position": record["position"],
                    "weights": record["weights"], "iteration": record["iteration"],
                    "wolf": record["wolf"]})
    leaders.sort(key=lambda row: (row["fitness"], row["iteration"], row["wolf"]))
    del leaders[3:]


def run(args):
    if args.population < 3 or args.iterations < 1 or args.candidate_epochs < 1:
        raise ValueError("Need at least 3 wolves, 1 iteration and 1 training epoch")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.dataset_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "profile": args.profile,
        "population": args.population, "iterations": args.iterations,
        "candidate_epochs": args.candidate_epochs, "batch_size": args.batch_size,
        "search_seed": args.search_seed, "candidate_seed": args.candidate_seed,
        "task_weights_ROP_RS": list(TASK_WEIGHTS),
        "objective": "0.9 validation MAPE_ROP + 0.1 validation MAPE_RS in original units",
        "search_variables": "(lambda_Huber, lambda_Quantile) in [0,1]^2; sum<=1; lambda_MAE=1-sum",
        "constraint_repair": "clip each search variable to [0,1], then divide by sum if sum>1",
        "a_schedule": "2 - 2*t/(iterations-1); update t=0,...,iterations-2; final population only evaluated",
        "candidate_training": "fresh CapacityMSAFN, fixed seed, Adam lr 0.001, fixed epochs, no early stop",
        "dataset_sha256": data["dataset_sha256"],
        "manifest_sha256": data["manifest_sha256"],
        "split_sha256": data["split_sha256"],
        "selection_bias_from_preprocessing": data["protocol"]["selection_bias"],
        "test_used_in_fitness": False,
        "device": device, "torch_version": torch.__version__,
    }
    config_path, state_path = args.output / "config.json", args.output / "state.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError("Existing run configuration differs; use a fresh output folder")
    else:
        save_json_atomic(config_path, config)
    rng = np.random.default_rng(args.search_seed)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        rng.bit_generator.state = state["rng_state"]
    else:
        state = initialize_state(rng, args.population)
        save_json_atomic(state_path, state)
    if state["iteration"] >= args.iterations and (args.output / "result.json").exists():
        print("ALREADY_COMPLETE", (args.output / "result.json").read_text(encoding="utf-8"), flush=True)
        return
    started = time.time()
    while state["iteration"] < args.iterations:
        iteration = state["iteration"]
        population = np.asarray(state["population"], dtype=np.float64)
        for wolf in range(state["next_wolf"], args.population):
            position = repair_triangle(population[wolf])
            weights = coefficients(position)
            fitness, mape, model_state = fit_candidate(
                weights, data, args.candidate_epochs, args.candidate_seed,
                args.batch_size, device)
            record = {
                "iteration": iteration, "wolf": wolf,
                "position": position.tolist(), "weights": weights,
                "fitness": fitness, "validation_MAPE_ROP": mape[0],
                "validation_MAPE_RS": mape[1],
            }
            prior_best = state["leaders"][0]["fitness"] if state["leaders"] else float("inf")
            if fitness < prior_best:
                torch.save({
                    "model_class": "CapacityMSAFN", "model_config": data["protocol"]["architecture"],
                    "model_state": model_state,
                    "scalers": data["checkpoint"]["scalers"],
                    "split_sha256": data["split_sha256"],
                    "weights": weights, "task_weights": TASK_WEIGHTS,
                    "fitness": fitness, "validation_MAPE": mape,
                    "iteration": iteration, "wolf": wolf,
                    "dataset_sha256": data["dataset_sha256"],
                    "search_profile": args.profile,
                }, args.output / "best_candidate.pt")
            update_leaders(state["leaders"], record)
            state["evaluations"].append(record)
            state["next_wolf"] = wolf + 1
            state["rng_state"] = rng.bit_generator.state
            save_json_atomic(state_path, state)
            print("EVALUATION", json.dumps(record), flush=True)
        if iteration < args.iterations - 1:
            a = 2.0 * (1.0 - iteration / (args.iterations - 1))
            population = move_wolves(population, state["leaders"], a, rng)
            state["population"] = population.tolist()
        state["iteration"] += 1
        state["next_wolf"] = 0
        state["rng_state"] = rng.bit_generator.state
        save_json_atomic(state_path, state)
        print("ITERATION_COMPLETE", iteration + 1,
              "best_fitness", state["leaders"][0]["fitness"], flush=True)
    result = {
        "profile": args.profile, "completed": True,
        "evaluations": len(state["evaluations"]),
        "best": state["leaders"][0],
        "seconds_this_invocation": time.time() - started,
        "test_evaluated": False,
        "interpretation": "Smoke result is a code check, not the paper's 30x100 GWO estimate"
        if args.profile == "smoke" else "Full configured search completed",
    }
    save_json_atomic(args.output / "result.json", result)
    print("COMPLETE", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--candidate-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--candidate-seed", type=int, default=42)
    args = parser.parse_args()
    defaults = PROFILES[args.profile]
    args.population = defaults[0] if args.population is None else args.population
    args.iterations = defaults[1] if args.iterations is None else args.iterations
    args.candidate_epochs = defaults[2] if args.candidate_epochs is None else args.candidate_epochs
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
