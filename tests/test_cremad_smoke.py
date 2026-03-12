import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
import wave

import numpy as np
from PIL import Image
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CREMAD_MODULE_PATH = REPO_ROOT / "src" / "synib" / "mydatasets" / "Irony_Cremad" / "dataloader.py"


def _load_cremad_module():
    sys.modules["torchaudio"] = _build_torchaudio_stub()
    sys.modules["librosa"] = _build_librosa_stub()
    spec = importlib.util.spec_from_file_location("cremad_dataset_module", CREMAD_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_torchaudio_stub():
    module = types.ModuleType("torchaudio")

    def load(path):
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return torch.from_numpy(audio).unsqueeze(0), sample_rate

    class _Functional:
        @staticmethod
        def resample(audio, orig_freq, new_freq):
            if orig_freq == new_freq:
                return audio
            raise NotImplementedError("Smoke test stub only supports matching sample rates")

    module.load = load
    module.functional = _Functional()
    return module


def _build_librosa_stub():
    module = types.ModuleType("librosa")

    def load(path, sr=22050):
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_rate != sr:
            raise NotImplementedError("Smoke test stub only supports matching sample rates")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return audio, sample_rate

    def stft(samples, n_fft=512, hop_length=353):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape[0] < n_fft:
            samples = np.pad(samples, (0, n_fft - samples.shape[0]))
        frames = []
        for start in range(0, max(samples.shape[0] - n_fft + 1, 1), hop_length):
            frame = samples[start : start + n_fft]
            if frame.shape[0] < n_fft:
                frame = np.pad(frame, (0, n_fft - frame.shape[0]))
            frames.append(np.fft.rfft(frame))
        if not frames:
            frames.append(np.fft.rfft(samples[:n_fft]))
        return np.stack(frames, axis=1)

    module.load = load
    module.stft = stft
    return module


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def _write_wav(path: pathlib.Path, sr: int = 22050, seconds: float = 0.25, freq: float = 220.0) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = 0.25 * np.sin(2 * np.pi * freq * t)
    pcm = np.int16(samples * 32767)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())


def _make_sample(dataset_root: pathlib.Path, uid: str, color: int) -> None:
    audio_dir = dataset_root / "AudioWAV"
    image_dir = dataset_root / "Image-01-FPS" / uid
    face_dir = dataset_root / "Face_features"

    audio_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    _write_wav(audio_dir / f"{uid}.wav")
    np.save(face_dir / f"{uid}.npy", np.full((4, 6), fill_value=float(color), dtype=np.float32))

    for idx in range(3):
        image = Image.new("RGB", (16, 16), color=(color, color, color))
        image.save(image_dir / f"{idx:05d}.jpg")


class CremadSmokeTest(unittest.TestCase):
    def test_cremad_dataloader_smoke(self):
        module = _load_cremad_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            dataset_root = tmp_path / "cremad_data"
            assets_root = tmp_path / "assets"
            metadata_root = assets_root / "metadata"
            metadata_root.mkdir(parents=True, exist_ok=True)

            split_payload = {
                "1": {
                    "train": ["1001_IEO_NEU_XX.wav-0"],
                    "val": ["1002_IEO_HAP_XX.wav-0"],
                    "test": ["1003_IEO_SAD_XX.wav-0"],
                }
            }
            (metadata_root / "data_splits_val.pkl").write_text(json.dumps(split_payload), encoding="utf-8")
            (metadata_root / "normalization_audio.pkl").write_text(
                json.dumps({"total": {"mean": 0.0, "std": 1.0}}),
                encoding="utf-8",
            )

            _make_sample(dataset_root, "1001_IEO_NEU_XX", 32)
            _make_sample(dataset_root, "1002_IEO_HAP_XX", 96)
            _make_sample(dataset_root, "1003_IEO_SAD_XX", 160)

            config = AttrDict()
            config.dataset = AttrDict(
                {
                    "data_roots": str(dataset_root),
                    "sampling_rate": 22050,
                    "num_frame": 3,
                    "fps": 1,
                    "norm": False,
                    "assets_root": str(assets_root),
                    "return_data": {"video": True, "spectrogram": True, "audio": False, "face": False},
                    "data_split": {"method": "non_inclusive", "fold": 0},
                    "norm_wav_path": str(assets_root / "norms" / "wav_norm_fold0.pkl"),
                }
            )
            config.training_params = AttrDict(
                {
                    "seed": 0,
                    "batch_size": 2,
                    "test_batch_size": 2,
                    "data_loader_workers": 0,
                    "pin_memory": False,
                }
            )

            previous_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                dataloader = module.IronyCremadDataloader(config)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(len(dataloader.train_loader.dataset), 1)
            self.assertEqual(len(dataloader.valid_loader.dataset), 1)
            self.assertEqual(len(dataloader.test_loader.dataset), 1)

            batch = next(iter(dataloader.train_loader))
            self.assertIn("data", batch)
            self.assertEqual(tuple(batch["label"].shape), (1,))
            self.assertEqual(batch["label"].item(), 0)
            self.assertEqual(batch["data"][0].ndim, 3)
            self.assertEqual(batch["data"][0].shape[0], 1)
            self.assertGreater(batch["data"][0].shape[1], 0)
            self.assertGreater(batch["data"][0].shape[2], 0)
            self.assertEqual(tuple(batch["data"][1].shape), (1, 3, 3, 224, 224))
            self.assertNotIn(2, batch["data"])
            self.assertNotIn(3, batch["data"])
            self.assertTrue((assets_root / "norms" / "wav_norm_fold0.pkl").exists())


if __name__ == "__main__":
    unittest.main()
