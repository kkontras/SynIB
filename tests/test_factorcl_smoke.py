import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _make_split(num_samples=3, seq_len=4, vision_dim=6, audio_dim=5, text_dim=8):
    base = {
        "vision": np.random.randn(num_samples, seq_len, vision_dim).astype(np.float32),
        "audio": np.random.randn(num_samples, seq_len, audio_dim).astype(np.float32),
        "text": np.random.randn(num_samples, seq_len, text_dim).astype(np.float32),
        "labels": np.random.randint(0, 2, size=(num_samples, 1)).astype(np.int64),
    }
    base["text"][:, 0, :] = 1.0
    return base


class _ConfigNode(dict):
    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict) and not isinstance(value, _ConfigNode):
            value = _ConfigNode(value)
            self[item] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def _build_config(data_path, dataloader_class, data_type="mosi", batch_size=2):
    return _ConfigNode(
        {
            "task": "classification",
            "dataset": {
                "dataloader_class": dataloader_class,
                "data_roots": str(data_path),
                "data_type": data_type,
                "train_shuffle": False,
                "max_pad": False,
                "max_seq_len": 8,
                "flatten_time_series": False,
                "z_norm": False,
                "mustard_video_filter": {
                    "enabled": True,
                    "vision_absmax_threshold": 1000.0,
                    "check_aligned_segment": True,
                    "min_aligned_len": 1,
                },
            },
            "training_params": {
                "seed": 0,
                "batch_size": batch_size,
                "data_loader_workers": 0,
            },
        }
    )


class FactorCLSmokeTest(unittest.TestCase):
    def test_factorcl_dataloader_batch(self):
        from synib.mydatasets.Factor_CL_Datasets import FactorCL_Dataloader

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "mosi_data.pkl"
            payload = {"train": _make_split(), "valid": _make_split(), "test": _make_split()}
            with data_path.open("wb") as f:
                pickle.dump(payload, f)

            loader = FactorCL_Dataloader(_build_config(data_path, "FactorCL_Dataloader"))
            batch = next(iter(loader.train_loader))
            self.assertIn("data", batch)
            self.assertEqual(set(batch["data"].keys()), {"c", "f", "g"})
            self.assertEqual(batch["data"]["c"].shape[0], 2)
            self.assertEqual(batch["label"].dtype, torch.int64)

    def test_mustard_filtered_dataloader_drops_corrupt_sample(self):
        from synib.mydatasets.Factor_CL_Datasets import FactorCL_MustardFiltered_Dataloader

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "mustard_data.pkl"
            payload = {"train": _make_split(), "valid": _make_split(), "test": _make_split()}
            payload["train"]["vision"][0, 1:, :] = 1e7
            with data_path.open("wb") as f:
                pickle.dump(payload, f)

            loader = FactorCL_MustardFiltered_Dataloader(
                _build_config(data_path, "FactorCL_MustardFiltered_Dataloader", data_type="sarcasm")
            )
            self.assertEqual(len(loader.train_loader.dataset), 2)
            batch = next(iter(loader.train_loader))
            self.assertEqual(set(batch["data"].keys()), {"c", "f", "g"})

    def test_downloader_local_file_materialization(self):
        from synib.mydatasets.Factor_CL_Datasets.downloaders.download_mosi import download_mosi

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "source.pkl"
            with source.open("wb") as f:
                pickle.dump({"ok": True}, f)

            output = download_mosi(tmpdir / "prepared", "unused", "mosi_data.pkl", local_file=source)
            self.assertTrue(output.exists())
            with output.open("rb") as f:
                self.assertEqual(pickle.load(f), {"ok": True})


if __name__ == "__main__":
    unittest.main()
