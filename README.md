# MSAFN reproducibility package: fixed 10,853-row dataset

This package contains the optimized four-branch `CapacityMSAFN` model, the three-component robust loss, Grey Wolf Optimization (GWO) for its loss weights, and three **preassigned** datasets. Training and GWO directly load `data/train.pt.gz`, `data/validation.pt.gz`, and `data/test.pt.gz`. They do not create or change the split.

## Files

| File | Purpose |
| --- | --- |
| `data/train.pt.gz` | 5,366 training observations |
| `data/validation.pt.gz` | 1,686 validation observations |
| `data/test.pt.gz` | 3,801 held-out test observations |
| `data/manifest.json` | SHA-256 hashes, sample counts, source hashes, tensor schema, architecture settings |
| `data/*_source_rows.csv` | Row correspondence to the filtered 10,853-row file and original source file |
| `m.py`, `m_local_tuned.py`, `m_structural.py`, `m_capacity.py` | Network and loss definitions |
| `fixed_dataset.py` | Hash-checked loading of the three fixed datasets |
| `gwo_composite_search.py` | Outer-loop GWO candidate search; loads **only train and validation** |
| `train_final.py` | Train with selected weights, then evaluate the held-out test set once |
| `train.py` | Original MSAFN baseline entry point and shared preprocessing/evaluation utilities; now uses fixed files |
| `test_*.py` | Structural, loss, data-loading and GWO invariants |

Each `.pt.gz` decompresses to a PyTorch dictionary with four float tensor keys. Per-sample shapes are `construction [6,30]`, `surrounding rock [5]`, `mwd [7,75,9]`, and `out [9]`. The model receives historical MWD `mwd[...,1:]` (eight channels), construction transposed to `[30,6]`, surrounding rock `[5]`, and five current active MWD values `out[:,[2,3,4,5,6]]`. The targets are `out[:,[1,7]]` in `(ROP,RS)` order. Current derived MSE is excluded as an input. Featurewise min-max scaling is fit on `train` only; validation/test are transformed using those training parameters without clipping.

## Setup and quick verification

Use Python 3.9.20 or a compatible Python version. The checked environment used PyTorch 2.5.1 and NumPy 1.26.4 on Windows with an RTX 3060. Install dependencies for your platform; choose the appropriate PyTorch wheel for your CUDA/CPU environment. Then run from this directory:

```bash
python -m unittest test_msafn.py test_joint_loss.py test_gwo_composite_search.py
python gwo_composite_search.py --profile smoke --output runs/gwo_smoke
```

The smoke profile is **4 wolves × 2 iterations × 2 training epochs**, totaling eight candidate trainings. It is an execution check only. It does not substantiate the manuscript's reported optimized weights or final predictive performance.

For a short end-to-end check of the final-training script with explicitly supplied weights, run:

```bash
python train_final.py --weights 0.424 0.357 0.219 --epochs 1 --output runs/final_smoke
```

The one-epoch test metrics from this command are only a code check, not a paper result. `0.424/0.357/0.219` are the **manuscript-reported** coefficients; they have not been verified as the output of a completed GWO search in this package.

## Paper-profile optimization and final evaluation

The paper profile is configured for 30 wolves, 100 GWO iterations and 100 fixed training epochs per candidate, amounting to 3,000 independent candidate trainings. **The full profile has not been executed for this package.** Only run it if the intended manuscript experiment requires it:

```bash
python gwo_composite_search.py --profile paper --output runs/gwo_paper
python train_final.py --gwo-result runs/gwo_paper/result.json --output runs/final_paper
```

The GWO code searches `(λ_Huber, λ_Quantile)` with each coordinate in `[0,1]` and their sum at most 1; `λ_MAE=1−λ_Huber−λ_Quantile`. An infeasible candidate is clipped to `[0,1]` and, if needed, its first two weights are normalized by their sum. Initial triples come from a uniform Dirichlet distribution. The standard alpha/beta/delta encircling update is used with `a_t=2−2t/(T−1)` for the 99 updates preceding the final population evaluation. Search seed is 2026; each candidate is independently initialized and trained with seed 42, Adam learning rate 0.001, batch size 128, gradient clipping 5, and no early stopping. `state.json` records a restartable search after each candidate.

For each task, the normalized-target training loss is `λ_Huber Huber(δ=1.5) + λ_Quantile Quantile(τ=0.75) + λ_MAE MAE`. The two task losses are aggregated as `0.9 L_ROP + 0.1 L_RS`. Quantile error is `true−prediction`, so `τ=0.75` penalizes **underprediction** more strongly. Each candidate's fitness is `0.9 MAPE_ROP + 0.1 MAPE_RS` on the inverse-scaled validation data. True zeros, if any, are omitted only from that task's MAPE denominator. The test file is not opened by the GWO entry point. `train_final.py` opens the test file after selecting weights and completing the final training.

## Provenance and limits

The source is the existing `train3404` variant after error-based removal of 3,404 records from the **training portion**, leaving 10,853 records. `data/manifest.json` identifies the source dataset and source checkpoint by SHA-256, and the row maps record exact membership. The export reproduced the existing membership; it did not select new rows. The three output data hashes are checked on every load.

The original tensor export lacks per-record tunnel tube, excavation-cycle ID, and chainage metadata. Consequently, this package **cannot independently demonstrate** the manuscript's claimed spatially continuous, cycle-isolated train/validation/test division. This limitation must be reconciled with the original field metadata before making that claim to reviewers. Error-based exclusion of training records also changes the sampled population and should be disclosed in the manuscript.

The package is local and unpublished. Authors should confirm data ownership, permissions and any project confidentiality restrictions before granting external access. There is no reviewer-accessible repository URL or dataset DOI in this package.
