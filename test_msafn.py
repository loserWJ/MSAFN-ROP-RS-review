import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from m import MSAFN, CompositeRobustLoss, PositionalEncoding, Attention
from train import CompleteBatchSampler, fit_scalers, transform, load_legacy_data
from fixed_dataset import sha256


torch.set_num_threads(2)


def inputs(batch=3):
    return (torch.rand(batch, 7, 75, 8), torch.rand(batch, 30, 6),
            torch.rand(batch, 5), torch.rand(batch, 5))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_dimensions_and_independent_towers(self):
        model = MSAFN().eval()
        with torch.no_grad():
            f = model.forward_features(*inputs())
        expected = dict(csan=32, transformer=16, construction_dnn=16,
                        current=14, context=64, latent=12, tower_input=26)
        self.assertEqual({k: tuple(v.shape) for k, v in f.items()},
                         {k: (3, size) for k, size in expected.items()})
        self.assertFalse({id(p) for p in model.tower_a.parameters()} &
                         {id(p) for p in model.tower_b.parameters()})
        self.assertEqual(model.mwdt.depth_lstm.hidden_size, 32)
        for tower in (model.tower_a, model.tower_b):
            self.assertEqual([(l.in_features, l.out_features) for l in tower.modules()
                              if isinstance(l, nn.Linear)], [(26, 16), (16, 6), (6, 1)])
        self.assertTrue(torch.equal(f["tower_input"][:, 12:], f["current"]))

    def test_batch_isolation_and_layout(self):
        model = MSAFN().eval()
        x = inputs()
        with torch.no_grad():
            together = torch.cat(model(*x), dim=1)
            alone = torch.cat(model(*(v[:1] for v in x)), dim=1)
            paper = torch.cat(model(x[0].permute(0, 1, 3, 2), *x[1:]), dim=1)
            reordered = torch.cat(model(*(v[[2, 0, 1]] for v in x)), dim=1)
        torch.testing.assert_close(together[:1], alone, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(together, paper)
        torch.testing.assert_close(together[[2, 0, 1]], reordered, atol=2e-6, rtol=1e-5)
        self.assertTrue(((together > 0) & (together < 1)).all())

    def test_all_branches_have_finite_nonzero_gradients(self):
        model = MSAFN()
        prediction = torch.cat(model(*inputs()), dim=1)
        CompositeRobustLoss()(prediction, torch.rand(3, 2)).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for name in ("mwdt", "construction", "dnn1_blast", "dnn2_blast",
                     "context_fusion", "enhancer", "final_cross", "fusion", "tower_a", "tower_b"):
            norm = sum(float(p.grad.abs().sum()) for p in getattr(model, name).parameters())
            self.assertGreater(norm, 0, name)
        # Q/K gradients prove attention isn't a single-token softmax constant.
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                for projection in (module.q, module.k):
                    self.assertGreater(float(projection.weight.grad.abs().sum()), 0, name)

    def test_spatial_attention_axes(self):
        model = MSAFN().eval()
        shapes = {}
        handles = []
        for name in ("depth_self", "lateral_self", "lateral_to_depth", "depth_to_lateral"):
            def hook(_module, args, key=name):
                shapes[key] = [tuple(a.shape) for a in args]
            handles.append(getattr(model.mwdt, name).register_forward_pre_hook(hook))
        with torch.no_grad():
            model.mwdt(inputs(2)[0])
        for h in handles:
            h.remove()
        self.assertEqual(shapes["depth_self"], [(14, 75, 64)] * 2)
        self.assertEqual(shapes["lateral_self"], [(150, 7, 8)] * 2)
        self.assertEqual(shapes["lateral_to_depth"], [(14, 75, 64)] * 2)
        self.assertEqual(shapes["depth_to_lateral"], [(150, 7, 64)] * 2)

    def test_position_encoding_varies_by_position_not_batch(self):
        y = PositionalEncoding()(torch.zeros(3, 30, 4))
        torch.testing.assert_close(y[0], y[1])
        self.assertFalse(torch.equal(y[0, 0], y[0, 1]))

    def test_joint_mode_and_checkpoint(self):
        model = MSAFN(current_input_mode="joint10").eval()
        self.assertEqual(model.dnn1_blast.layers[0].in_features, 10)
        self.assertEqual(model.dnn2_blast.layers[0].in_features, 10)
        x = inputs(2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save({"config": model.config, "state": model.state_dict()}, path)
            saved = torch.load(path, weights_only=True)
            restored = MSAFN(**saved["config"]).eval()
            restored.load_state_dict(saved["state"])
            with torch.no_grad():
                for a, b in zip(model(*x), restored(*x)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_input_validation(self):
        model = MSAFN().eval()
        x = inputs()
        with self.assertRaises(ValueError):
            model(*x[:3], torch.rand(3, 6))
        with self.assertRaises(ValueError):
            model(x[0], x[1].transpose(1, 2), *x[2:])

    def test_loss_definition_and_conflicting_direction(self):
        y = torch.ones(2, 2)
        exact = CompositeRobustLoss()
        self.assertEqual(float(exact(y, y)), 0)
        # Equal magnitude errors of 1: Huber=.5, MAE=1, quantile=.75/.25.
        self.assertAlmostEqual(float(exact(y - 1, y)), .424 * .5 + .357 * .75 + .219, places=6)
        self.assertAlmostEqual(float(exact(y + 1, y)), .424 * .5 + .357 * .25 + .219, places=6)
        prose = CompositeRobustLoss(quantile_mode="overprediction")
        self.assertGreater(float(prose(y + 1, y)), float(prose(y - 1, y)))

    def test_scalers_no_test_leakage_and_constant_features(self):
        x = torch.tensor([[1., 4.], [3., 4.], [100., 4.]])
        scalers = fit_scalers((x,), torch.tensor([0, 1]))
        y, = transform((x,), scalers)
        torch.testing.assert_close(y[:2], torch.tensor([[0., 0.], [1., 0.]]))
        self.assertGreater(float(y[-1, 0]), 1.)
        self.assertTrue(torch.isfinite(y).all())

    def test_complete_batches(self):
        for n in (2, 3, 32, 33, 65):
            sampler = CompleteBatchSampler(n, 32, 42)
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            self.assertTrue(all(len(b) >= 2 for b in batches))
            self.assertEqual(sorted(i for b in batches for i in b), list(range(n)))

    def test_training_cli_and_column_mapping(self):
        # Synthetic end-to-end fixture only, never a paper accuracy experiment.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            n = 40
            raw = {
                "construction": torch.rand(n, 6, 30),
                "surrounding rock": torch.rand(n, 5),
                "mwd": torch.rand(n, 7, 75, 9),
                "out": torch.rand(n, 9),
            }
            torch.save(raw, root / "data.pt")
            adapted = load_legacy_data(root / "data.pt")
            torch.testing.assert_close(adapted[3], raw["out"][:, [2, 3, 4, 5, 6]])
            torch.testing.assert_close(adapted[4], raw["out"][:, [1, 7]])
            folder = root / "data"
            folder.mkdir()
            records = {}
            for name, rows in (("train", slice(0, 28)),
                               ("validation", slice(28, 34)),
                               ("test", slice(34, 40))):
                path = folder / f"{name}.pt.gz"
                with gzip.open(path, "wb") as stream:
                    torch.save({key: value[rows] for key, value in raw.items()}, stream)
                records[name] = {"file": path.name, "rows": len(raw["out"][rows]),
                                 "sha256": sha256(path)}
            (folder / "manifest.json").write_text(
                json.dumps({"total_rows": n, "splits": records}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("train.py")),
                 "--dataset-dir", str(folder),
                 "--output", str(root / "run"), "--epochs", "1",
                 "--device", "cpu", "--batch-size", "8"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = torch.load(root / "run" / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["data_columns"]["targets"], [1, 7])
            with (root / "run" / "test_predictions.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            metrics = json.loads((root / "run" / "metrics.json").read_text())
            self.assertEqual(set(metrics), {"ROP", "RS"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
