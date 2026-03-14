"""
Smoke tests for MUStARD_Dataset.py.

Mocks cv2 so no real video files are needed; uses 20 synthetic metadata entries.
Verifies:
  1. Item schema matches the plan (data/label/id with correct sub-keys and dtypes).
  2. speaker2id is consistent across all three splits.
  3. Splits are non-overlapping and together cover all samples.
  4. collate_mustard_raw produces correctly-shaped tensors.
  5. MUStARD_Raw_Dataloader can be constructed and iterated.
"""

import importlib.util
import json
import pathlib
import sys
import types
import unittest

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MUSTARD_MODULE_PATH = (
    REPO_ROOT / "src" / "synib" / "mydatasets" / "MUStARD" / "MUStARD_Dataset.py"
)

# ---------------------------------------------------------------------------
# cv2 stub  — returns 3 synthetic 16×16 BGR frames per video
# ---------------------------------------------------------------------------

_N_FAKE_FRAMES = 3
_FAKE_FPS = 1.0
_FAKE_H, _FAKE_W = 16, 16


def _build_cv2_stub() -> types.ModuleType:
    module = types.ModuleType("cv2")
    module.CAP_PROP_FPS = 5
    module.CAP_PROP_FRAME_COUNT = 6
    module.COLOR_BGR2RGB = 4

    class _FakeCap:
        def __init__(self, path: str) -> None:
            self._current = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == module.CAP_PROP_FPS:
                return _FAKE_FPS
            if prop == module.CAP_PROP_FRAME_COUNT:
                return float(_N_FAKE_FRAMES)
            return 0.0

        def read(self):
            if self._current >= _N_FAKE_FRAMES:
                return False, None
            frame = np.full(
                (_FAKE_H, _FAKE_W, 3),
                fill_value=self._current * 80,
                dtype=np.uint8,
            )
            self._current += 1
            return True, frame

        def release(self) -> None:
            pass

    module.VideoCapture = _FakeCap

    def cvtColor(frame, code):  # noqa: N802
        return frame  # stub: identity (colour layout doesn't matter for shapes)

    def resize(frame, dsize):  # noqa: N802
        h, w = dsize[1], dsize[0]
        return np.zeros((h, w, 3), dtype=np.uint8)

    module.cvtColor = cvtColor
    module.resize = resize
    return module


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load_mustard_module():
    """Load MUStARD_Dataset.py with cv2 replaced by the stub."""
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = _build_cv2_stub()
    spec = importlib.util.spec_from_file_location(
        "mustard_dataset_module", MUSTARD_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


# 20 synthetic samples — balanced sarcasm labels, two speakers
_METADATA = [
    {
        "id": f"clip_{i:03d}",
        "utterance": f"utterance number {i}",
        "speaker": "SHELDON" if i % 2 == 0 else "LEONARD",
        "sarcasm": i % 2 == 0,
        "context": [f"context turn {i}"],
        "context_speakers": ["LEONARD"],
        "show": "BBT",
    }
    for i in range(20)
]


def _make_config(tmp_path: pathlib.Path, batch_size: int = 2) -> AttrDict:
    mustard_root = tmp_path / "mustard_raw"
    videos_dir = mustard_root / "videos"
    videos_dir.mkdir(parents=True)

    metadata_path = mustard_root / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(_METADATA, f)

    # Create empty placeholder .mp4 files (cv2 is mocked, so content is irrelevant)
    for rec in _METADATA:
        (videos_dir / f"{rec['id']}.mp4").write_bytes(b"")

    config = AttrDict()
    config.dataset = AttrDict(
        data_roots=str(metadata_path),
        fps=1,
        image_size=224,
        val_split_rate=0.15,
        test_split_rate=0.15,
    )
    config.training_params = AttrDict(batch_size=batch_size, seed=42)
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMUStARDDatasetItemSchema(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.module = _load_mustard_module()
        self.config = _make_config(self.tmp_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_item_keys_present(self):
        ds = self.module.MUStARD_RawDataset(config=self.config, split="train")
        self.assertGreater(len(ds), 0)
        item = ds[0]

        self.assertIn("data", item)
        self.assertIn("label", item)
        self.assertIn("id", item)

        data = item["data"]
        for key in ("text", "speaker", "speaker_id", "video_frames", "context"):
            self.assertIn(key, data, msg=f"Missing key: {key!r}")

    def test_item_types(self):
        ds = self.module.MUStARD_RawDataset(config=self.config, split="train")
        item = ds[0]
        data = item["data"]

        self.assertIsInstance(data["text"], str)
        self.assertIsInstance(data["speaker"], str)
        self.assertIsInstance(data["speaker_id"], int)
        self.assertIsInstance(data["context"], list)
        self.assertIsInstance(item["id"], str)

    def test_video_frames_shape_and_dtype(self):
        ds = self.module.MUStARD_RawDataset(config=self.config, split="train")
        frames = ds[0]["data"]["video_frames"]

        self.assertEqual(frames.ndim, 4, "Expected (F, C, H, W)")
        self.assertEqual(frames.shape[1], 3, "Expected 3 channels")
        self.assertEqual(frames.shape[2], 224)
        self.assertEqual(frames.shape[3], 224)
        self.assertEqual(frames.dtype, torch.float32)
        self.assertGreaterEqual(frames.min().item(), 0.0)
        self.assertLessEqual(frames.max().item(), 1.0)

    def test_label_dtype_and_range(self):
        ds = self.module.MUStARD_RawDataset(config=self.config, split="train")
        label = ds[0]["label"]
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.dtype, torch.long)
        self.assertIn(label.item(), [0, 1])


class TestMUStARDSplits(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.module = _load_mustard_module()
        self.config = _make_config(self.tmp_path)

    def tearDown(self):
        self._tmp.cleanup()

    def _get_split(self, split):
        return self.module.MUStARD_RawDataset(config=self.config, split=split)

    def test_speaker2id_consistent_across_splits(self):
        ds_train = self._get_split("train")
        ds_val = self._get_split("val")
        ds_test = self._get_split("test")
        self.assertEqual(ds_train.speaker2id, ds_val.speaker2id)
        self.assertEqual(ds_train.speaker2id, ds_test.speaker2id)

    def test_speaker2id_covers_all_speakers(self):
        ds = self._get_split("train")
        self.assertIn("SHELDON", ds.speaker2id)
        self.assertIn("LEONARD", ds.speaker2id)

    def test_splits_are_disjoint(self):
        train_ids = {r["id"] for r in self._get_split("train").records}
        val_ids = {r["id"] for r in self._get_split("val").records}
        test_ids = {r["id"] for r in self._get_split("test").records}

        self.assertEqual(len(train_ids & val_ids), 0, "Train and val overlap")
        self.assertEqual(len(train_ids & test_ids), 0, "Train and test overlap")
        self.assertEqual(len(val_ids & test_ids), 0, "Val and test overlap")

    def test_splits_cover_all_samples(self):
        train_ids = {r["id"] for r in self._get_split("train").records}
        val_ids = {r["id"] for r in self._get_split("val").records}
        test_ids = {r["id"] for r in self._get_split("test").records}
        all_ids = train_ids | val_ids | test_ids
        self.assertEqual(len(all_ids), len(_METADATA))

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            self._get_split("invalid_split")


class TestCollate(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.module = _load_mustard_module()
        self.config = _make_config(self.tmp_path, batch_size=2)

    def tearDown(self):
        self._tmp.cleanup()

    def test_collate_shapes(self):
        ds = self.module.MUStARD_RawDataset(config=self.config, split="train")
        B = min(3, len(ds))
        items = [ds[i] for i in range(B)]
        batch = self.module.collate_mustard_raw(items)

        self.assertIn("data", batch)
        self.assertIn("label", batch)
        self.assertIn("id", batch)

        data = batch["data"]

        # text / speaker / context are lists of length B
        self.assertEqual(len(data["text"]), B)
        self.assertEqual(len(data["speaker"]), B)
        self.assertEqual(len(data["context"]), B)

        # speaker_id: (B,) int64
        self.assertEqual(tuple(data["speaker_id"].shape), (B,))
        self.assertEqual(data["speaker_id"].dtype, torch.long)

        # video_frames: (B, max_F, 3, H, W)
        vf = data["video_frames"]
        self.assertEqual(vf.ndim, 5)
        self.assertEqual(vf.shape[0], B)
        self.assertEqual(vf.shape[2], 3)
        self.assertEqual(vf.shape[3], 224)
        self.assertEqual(vf.shape[4], 224)
        self.assertEqual(vf.dtype, torch.float32)

        # attention_mask_video: (B, max_F)
        mask = data["attention_mask_video"]
        self.assertEqual(tuple(mask.shape), (B, vf.shape[1]))
        self.assertEqual(mask.dtype, torch.long)
        # All frames from a non-padded single-fps video should be attended
        self.assertTrue((mask >= 0).all() and (mask <= 1).all())

        # label: (B,)
        self.assertEqual(tuple(batch["label"].shape), (B,))
        self.assertEqual(batch["label"].dtype, torch.long)

        # id: list of B strings
        self.assertEqual(len(batch["id"]), B)

    def test_collate_pads_unequal_frames(self):
        """Manually build items with different frame counts and verify padding."""
        ds = self.module.MUStARD_RawDataset(config=self.config, split="train")
        item_long = ds[0]
        item_short = ds[0]

        # Override video_frames to have different lengths
        item_long = dict(item_long)
        item_long["data"] = dict(item_long["data"])
        item_short = dict(item_short)
        item_short["data"] = dict(item_short["data"])

        item_long["data"]["video_frames"] = torch.zeros(5, 3, 224, 224)
        item_short["data"]["video_frames"] = torch.zeros(2, 3, 224, 224)

        batch = self.module.collate_mustard_raw([item_long, item_short])
        vf = batch["data"]["video_frames"]
        mask = batch["data"]["attention_mask_video"]

        self.assertEqual(vf.shape[1], 5, "max_frames should be 5")
        self.assertEqual(tuple(mask[0].tolist()), (1, 1, 1, 1, 1))
        self.assertEqual(tuple(mask[1].tolist()), (1, 1, 0, 0, 0))


class TestDataloader(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.module = _load_mustard_module()
        self.config = _make_config(self.tmp_path, batch_size=2)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dataloaders_non_empty(self):
        dl = self.module.MUStARD_Raw_Dataloader(self.config)
        self.assertGreater(len(dl.train_loader.dataset), 0)
        self.assertGreater(len(dl.valid_loader.dataset), 0)
        self.assertGreater(len(dl.test_loader.dataset), 0)

    def test_train_loader_batch(self):
        dl = self.module.MUStARD_Raw_Dataloader(self.config)
        batch = next(iter(dl.train_loader))
        self.assertIn("data", batch)
        self.assertIn("label", batch)
        self.assertIn("id", batch)
        vf = batch["data"]["video_frames"]
        self.assertEqual(vf.ndim, 5)
        self.assertEqual(vf.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
