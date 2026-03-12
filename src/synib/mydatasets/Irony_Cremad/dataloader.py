import csv
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ASSETS_DIR = PACKAGE_DIR / "assets"
DEFAULT_METADATA_DIR = DEFAULT_ASSETS_DIR / "metadata"
DEFAULT_NORMS_DIR = DEFAULT_ASSETS_DIR / "norms"

BASE_CLASS_DICT = {"NEU": 0, "HAP": 1, "SAD": 2, "FEA": 3, "DIS": 4, "ANG": 5}
BASE_CLASS_DICT_INV = {0: "NEU", 1: "HAP", 2: "SAD", 3: "FEA", 4: "DIS", 5: "ANG", 6: "SAR"}


def _read_csv_rows(path: Path) -> List[List[str]]:
    with path.open(encoding="UTF-8-sig") as f:
        return list(csv.reader(f))


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _safe_pickle_load(path: Optional[str]) -> Optional[Any]:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.loads(f.read())
    return None


def _safe_pickle_save(path: str, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(pickle.dumps(obj))


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collate_fn_padd(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"data": {}}
    out["label"] = torch.LongTensor([b["label"] for b in batch])
    out["idx"] = torch.LongTensor([b["idx"] for b in batch])

    data = [b["data"] for b in batch]

    def cat_if_present(k: int) -> None:
        items = [d[k].unsqueeze(0) for d in data if d.get(k, False) is not False]
        if items:
            out["data"][k] = torch.cat(items, dim=0)

    cat_if_present(0)
    cat_if_present(1)

    audios = [d[2] for d in data if d.get(2, False) is not False]
    if audios:
        lengths = torch.LongTensor([len(a) for a in audios])
        out["data"][2] = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        mask = torch.zeros((len(audios), int(lengths.max())), dtype=torch.float32)
        for i, length in enumerate(lengths.tolist()):
            mask[i, :length] = 1.0
        out["data"]["attention_mask_audio"] = mask

    faces = [d[3] for d in data if d.get(3, False) is not False]
    if faces:
        lengths = [len(f) for f in faces]
        max_len = min(max(lengths), 150)
        faces = [f[:max_len] for f in faces]
        out["data"][3] = torch.nn.utils.rnn.pad_sequence(faces, batch_first=True)
        mask = torch.zeros((len(faces), max_len), dtype=torch.float32)
        for i, length in enumerate(lengths):
            mask[i, : min(length, max_len)] = 1.0
        out["data"]["attention_mask_face"] = mask

    return out


class IronyCremadDataset(Dataset):
    def __init__(self, config: Any, fps: int = 1, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger("IronyCremadDataset")

        ds = self.config.dataset
        self.num_frame = int(ds.get("num_frame", 3))
        self.norm_type = ds.get("norm_type", False)
        self.sampling_rate = int(ds.sampling_rate)
        self.fps = int(ds.get("fps", fps))
        self.return_data = ds.get(
            "return_data",
            {"video": True, "spectrogram": True, "audio": False, "face": False, "face_image": False},
        )

        self.ironic_rate = float(ds.get("ironic_rate", 0.0))
        self.ironic_label_name = str(ds.get("ironic_label_name", "IRONIC_MISMATCH"))
        self.ironic_modes = set(ds.get("ironic_modes", ["train"]))

        assets_root = Path(ds.get("assets_root", DEFAULT_ASSETS_DIR))
        self.assets_root = assets_root
        self.metadata_root = assets_root / "metadata"
        self.norms_root = assets_root / "norms"

        roots = Path(ds.data_roots)
        self.visual_root = roots
        self.audio_root = roots / "AudioWAV"
        self.face_root = roots / "Face_features"

        self.item: List[str] = []
        self.image: List[str] = []
        self.audio: List[str] = []
        self.faces: List[str] = []
        self.label: List[int] = []

        self.class_dict = dict(BASE_CLASS_DICT)
        self.ironic_label_id = len(self.class_dict)
        self.class_dict[self.ironic_label_name] = self.ironic_label_id

        split = ds.get("data_split", {"fold": 0})
        fold = int(split.get("fold", 0))
        method = split.get("method", "inclusive")

        if method == "inclusive":
            self._split_inclusive(mode)
        elif method == "non_inclusive":
            self._split_noninclusive(fold, mode)
        else:
            raise ValueError(f"config.dataset.data_split.method must be 'inclusive' or 'non_inclusive', got {method!r}")

        self.audio_src_index = np.arange(len(self.audio), dtype=np.int64)
        self.ironic_mask = np.zeros(len(self.audio), dtype=np.bool_)

        self._load_or_build_wav_norm()
        if ds.get("norm", True):
            self._load_or_build_face_norm()

        self._apply_ironic_pairs_if_enabled()
        self.video_transforms = {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(224, 224), antialias=True),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(224, 224), antialias=True),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        images = self._get_images(idx)
        audio_idx = int(self.audio_src_index[idx])
        audio = self._get_audio_from_index(audio_idx)
        face = self._get_face(idx)
        spec = self._get_spectrogram_from_index(audio_idx, audio)
        return {"data": {0: spec, 1: images, 2: audio, 3: face}, "label": int(self.label[idx]), "idx": idx}

    def _paths_from_id(self, uid: str) -> Tuple[Path, Path, Path]:
        audio_path = self.audio_root / f"{uid}.wav"
        video_path = self.visual_root / f"Image-{self.fps:02d}-FPS" / uid
        face_path = self.face_root / f"{uid}.npy"
        return audio_path, video_path, face_path

    @staticmethod
    def _valid_triplet(audio_path: Path, video_path: Path, face_path: Path) -> bool:
        return audio_path.exists() and video_path.exists() and face_path.exists()

    def _split_inclusive(self, mode: str) -> None:
        self.norm_audio = {"total": {"mean": -7.1276217, "std": 5.116028}}

        train_rows = _read_csv_rows(self.metadata_root / "train.csv")
        test_rows = _read_csv_rows(self.metadata_root / "test.csv")

        train = self._collect_from_rows(train_rows)
        test = self._collect_from_rows(test_rows)
        total = {k: train[k] + test[k] for k in train.keys()}

        x = np.array([total["item"], total["image"], total["audio"], total["faces"]]).T
        y = np.array(total["label"])
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x,
            y,
            test_size=self.config.dataset.get("val_split_rate", 0.1),
            random_state=self.config.training_params.seed,
            stratify=y,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval,
            y_trainval,
            test_size=self.config.dataset.get("val_split_rate", 0.1),
            random_state=self.config.training_params.seed,
            stratify=y_trainval,
        )

        if mode == "train":
            x_mode, y_mode = x_train, y_train
        elif mode == "val":
            x_mode, y_mode = x_val, y_val
        elif mode == "test":
            x_mode, y_mode = x_test, y_test
        else:
            raise ValueError(f"mode must be one of train/val/test, got {mode!r}")

        self.item = x_mode[:, 0].tolist()
        self.image = x_mode[:, 1].tolist()
        self.audio = x_mode[:, 2].tolist()
        self.faces = x_mode[:, 3].tolist()
        self.label = y_mode.tolist()

    def _collect_from_rows(self, rows: List[List[str]]) -> Dict[str, List[Any]]:
        out = {"item": [], "image": [], "audio": [], "faces": [], "label": []}
        for uid, cls in rows:
            audio_path, video_path, face_path = self._paths_from_id(uid)
            if not self._valid_triplet(audio_path, video_path, face_path):
                continue
            out["item"].append(uid)
            out["image"].append(str(video_path))
            out["audio"].append(str(audio_path))
            out["faces"].append(str(face_path))
            out["label"].append(BASE_CLASS_DICT[cls])
        return out

    def _split_noninclusive(self, fold: int, mode: str) -> None:
        self.norm_audio = _read_json(self.metadata_root / "normalization_audio.pkl")
        splits = _read_json(self.metadata_root / "data_splits_val.pkl")

        for entry in splits[str(fold + 1)][mode]:
            name = entry.split("-")[0]
            label = BASE_CLASS_DICT[name.split("_")[2]]
            audio_path = self.audio_root / name
            video_path = self.visual_root / f"Image-{self.fps:02d}-FPS" / name.split(".")[0]
            face_path = self.face_root / name.replace(".wav", ".npy")
            if not self._valid_triplet(audio_path, video_path, face_path):
                continue
            self.item.append(name)
            self.image.append(str(video_path))
            self.audio.append(str(audio_path))
            self.faces.append(str(face_path))
            self.label.append(int(label))

    def _apply_ironic_pairs_if_enabled(self) -> None:
        if self.mode not in self.ironic_modes:
            return

        n = len(self.audio)
        if n < 2:
            return

        labels_np = np.asarray(self.label, dtype=np.int64)
        base_mask = labels_np != self.ironic_label_id
        base_labels = labels_np[base_mask]
        if base_labels.size == 0:
            return

        uniq, counts = np.unique(base_labels, return_counts=True)
        target = float(np.mean(counts))
        k = int(round(self.ironic_rate * target))
        k = max(0, min(k, int(base_mask.sum())))
        if k == 0:
            return

        idx_by_label: Dict[int, np.ndarray] = {}
        for label in uniq.tolist():
            idx_by_label[int(label)] = np.where(labels_np == label)[0]

        contradiction_map = {
            0: [],
            1: [2, 3, 4, 5],
            2: [1],
            3: [1],
            4: [1],
            5: [1],
        }

        candidates = np.where(base_mask)[0]
        candidates_t = torch.sort(torch.as_tensor(candidates, dtype=torch.long)).values
        generator = torch.Generator()
        generator.manual_seed(int(self.config.training_params.seed))
        chosen = candidates_t[torch.randperm(candidates_t.numel(), generator=generator)][:k].cpu().numpy()
        self.ironic_mask[chosen] = True

        for idx in chosen:
            src_label = int(labels_np[idx])
            preferred = contradiction_map.get(src_label, [])
            pool = [idx_by_label[label] for label in preferred if label in idx_by_label]
            if pool:
                donor_candidates = np.concatenate(pool)
                donor_candidates = donor_candidates[donor_candidates != idx]
            else:
                donor_candidates = np.where(labels_np != src_label)[0]
                donor_candidates = donor_candidates[donor_candidates != idx]

            if donor_candidates.size == 0:
                self.ironic_mask[idx] = False
                continue

            donor_t = torch.sort(torch.as_tensor(donor_candidates, dtype=torch.long)).values
            donor_generator = torch.Generator()
            donor_generator.manual_seed(int(self.config.training_params.seed + idx))
            donor_idx = int(donor_t[torch.randint(0, donor_t.numel(), (1,), generator=donor_generator)].item())
            self.audio_src_index[idx] = donor_idx
            self.label[idx] = self.ironic_label_id

    def _default_norm_path(self, name: str) -> str:
        return str(self.norms_root / name)

    def _load_or_build_wav_norm(self) -> None:
        path = self.config.dataset.get("norm_wav_path", self._default_norm_path("wav_norm.pkl"))
        norm = _safe_pickle_load(path)
        if norm is not None:
            self.wav_norm = norm
            self.logger.info("Loaded wav norm from %s", path)
            return
        if self.mode != "train":
            return
        self.wav_norm = self._compute_wav_norm()
        self.logger.warning("Saving wav norm to %s", path)
        _safe_pickle_save(path, self.wav_norm)

    def _compute_wav_norm(self) -> Dict[str, Any]:
        count = 0
        wav_sum = 0.0
        wav_sqsum = 0.0
        max_duration = 0.0
        for wav_path in tqdm(self.audio):
            audio, sample_rate = torchaudio.load(wav_path)
            audio = torchaudio.functional.resample(audio, sample_rate, self.sampling_rate)[0]
            max_duration = max(max_duration, float(audio.shape[0]) / float(self.sampling_rate))
            wav_sum += float(torch.sum(audio))
            wav_sqsum += float(torch.sum(audio ** 2))
            count += int(audio.numel())

        mean = wav_sum / count
        var = (wav_sqsum / count) - (mean ** 2)
        std = float(np.sqrt(var))
        return {"mean": mean, "std": std, "max_duration": max_duration}

    def _load_or_build_face_norm(self) -> None:
        path = self.config.dataset.get("norm_face_path", self._default_norm_path("face_norm.pkl"))
        norm = _safe_pickle_load(path)
        if norm is not None:
            self.face_norm = norm
            self.logger.info("Loaded face norm from %s", path)
            return
        if self.mode != "train":
            return
        self.face_norm = self._compute_face_norm()
        self.logger.warning("Saving face norm to %s", path)
        _safe_pickle_save(path, self.face_norm)

    def _compute_face_norm(self) -> Dict[str, Any]:
        count = 0
        feat_sum = 0.0
        feat_sqsum = 0.0
        max_faces = 0
        for face_path in tqdm(self.faces):
            feats = np.load(face_path, allow_pickle=True)
            feat_sum += np.sum(feats, axis=0)
            feat_sqsum += np.sum(feats ** 2, axis=0)
            count += feats.shape[0]
            max_faces = max(max_faces, feats.shape[0])

        mean = feat_sum / count
        var = (feat_sqsum / count) - (mean ** 2)
        std = np.sqrt(var)
        return {"mean": mean, "std": std, "max_faces": int(max_faces)}

    def _get_images(self, idx: int):
        if not self.return_data.get("video", False):
            return False
        transform = self.video_transforms[self.mode]
        frames = sorted(os.listdir(self.image[idx]))
        images = torch.zeros((self.num_frame, 3, 224, 224), dtype=torch.float32)
        for i, frame_name in enumerate(frames[: self.num_frame]):
            img = Image.open(os.path.join(self.image[idx], frame_name)).convert("RGB")
            images[i] = transform(img)
        return torch.permute(images, (1, 0, 2, 3))

    def _get_audio_from_index(self, audio_idx: int):
        if not self.return_data.get("audio", False):
            return False
        audio, sample_rate = torchaudio.load(self.audio[audio_idx])
        audio = torchaudio.functional.resample(audio, sample_rate, self.sampling_rate)[0]
        if hasattr(self, "wav_norm"):
            audio = (audio - float(self.wav_norm["mean"])) / float(self.wav_norm["std"])
        return audio

    def _get_face(self, idx: int):
        if not self.return_data.get("face", False):
            return False
        face = torch.from_numpy(np.load(self.faces[idx], allow_pickle=True))
        if hasattr(self, "face_norm"):
            mean = torch.as_tensor(self.face_norm["mean"])
            std = torch.as_tensor(self.face_norm["std"])
            face = (face - mean) / std
        return face

    def _get_spectrogram_from_index(self, audio_idx: int, audio):
        if not self.return_data.get("spectrogram", False):
            return False

        if audio is False:
            samples, _ = librosa.load(self.audio[audio_idx], sr=self.sampling_rate)
            resamples = np.tile(samples, 3)[: self.sampling_rate * 3]
        else:
            resamples = np.tile(audio.detach().cpu().numpy(), 3)[: self.sampling_rate * 3]
        resamples = np.clip(resamples, -1.0, 1.0)

        spec = librosa.stft(resamples, n_fft=512, hop_length=353)
        spec = np.log(np.abs(spec) + 1e-7)

        if self.norm_type == "per_sample":
            mean = float(np.mean(spec))
            std = float(np.std(spec))
            spec = (spec - mean) / (std + 1e-9)
        elif self.norm_type == "per_freq":
            mean = np.array(self.norm_audio["per_req"]["mean"])
            std = np.array(self.norm_audio["per_req"]["std"])
            spec = ((spec.T - mean) / (std + 1e-9)).T
        elif self.norm_type == "total":
            mean = float(self.norm_audio["total"]["mean"])
            std = float(self.norm_audio["total"]["std"])
            spec = (spec - mean) / (std + 1e-9)

        return torch.from_numpy(spec)


class IronyCremadDataloader:
    def __init__(self, config: Any):
        self.config = config
        train_ds, val_ds, test_ds = self._get_datasets()

        generator = torch.Generator()
        generator.manual_seed(int(self.config.training_params.seed))
        num_workers = int(getattr(self.config.training_params, "data_loader_workers", 0) or 0)

        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.config.training_params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
            generator=generator,
            collate_fn=collate_fn_padd,
            worker_init_fn=_seed_worker,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
            collate_fn=collate_fn_padd,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
            collate_fn=collate_fn_padd,
        )

    def _get_datasets(self):
        train_dataset = IronyCremadDataset(config=self.config, mode="train")
        valid_dataset = IronyCremadDataset(config=self.config, mode="val")
        test_dataset = IronyCremadDataset(config=self.config, mode="test")
        return train_dataset, valid_dataset, test_dataset
