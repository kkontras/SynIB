import csv
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
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


BASE_CLASS_DICT = {"NEU": 0, "HAP": 1, "SAD": 2, "FEA": 3, "DIS": 4, "ANG": 5}
BASE_CLASS_DICT_INV = {0: "NEU", 1: "HAP", 2: "SAD", 3: "FEA", 4: "DIS", 5: "ANG", 6: "SAR"}

# contradiction_map = {
#     0: [],
#     1: [2, 5, 4],
#     2: [1],
#     3: [1],
#     4: [1],
#     5: [0, 1, 2],
# }

def _read_csv_rows(path: str) -> List[List[str]]:
    with open(path, encoding="UTF-8-sig") as f:
        return list(csv.reader(f))
def _safe_pickle_load(path: Optional[str]) -> Optional[Any]:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.loads(f.read())
    return None
def _safe_pickle_save(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(pickle.dumps(obj))
def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _video_transform(mode: str, transform_dict) -> transforms.Compose:
    this_transform = transform_dict[mode]
    return this_transform
    # if mode == "train":
    #     return transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.RandomResizedCrop(224, antialias=True),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #         ]
    #     )
    # return transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Resize(size=(224, 224), antialias=True),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )

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
        for i, L in enumerate(lengths.tolist()):
            mask[i, :L] = 1.0
        out["data"]["attention_mask_audio"] = mask

    faces = [d[3] for d in data if d.get(3, False) is not False]
    if faces:
        lengths = [len(f) for f in faces]
        max_len = min(max(lengths), 150)
        faces = [f[:max_len] for f in faces]
        out["data"][3] = torch.nn.utils.rnn.pad_sequence(faces, batch_first=True)
        mask = torch.zeros((len(faces), max_len), dtype=torch.float32)
        for i, L in enumerate(lengths):
            mask[i, : min(L, max_len)] = 1.0
        out["data"]["attention_mask_face"] = mask

    return out

class CremadPlusDataset(Dataset):
    def __init__(self, config: Any, fps: int = 1, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.fps = fps
        self.logger = logging.getLogger('CREMADPlusDataset')


        ds = self.config.dataset
        self.num_frame = ds.get("num_frame", 3)
        self.norm_type = ds.get("norm_type", False)
        self.sampling_rate = ds.sampling_rate
        self.return_data = ds.get(
            "return_data",
            {"video": True, "spectrogram": True, "audio": False, "face": False, "face_image": False},
        )

        self.ironic_rate = float(ds.get("ironic_rate", 0.0))
        self.ironic_label_name = str(ds.get("ironic_label_name", "IRONIC_MISMATCH"))
        self.ironic_modes = set(ds.get("ironic_modes", ["train"]))

        roots = ds.data_roots
        self.visual_root = roots
        self.audio_root = os.path.join(roots, "AudioWAV")
        self.face_root = os.path.join(roots, "Face_features")

        self.item: List[str] = []
        self.image: List[str] = []
        self.audio: List[str] = []
        self.faces: List[str] = []
        self.label: List[int] = []

        self.class_dict = dict(BASE_CLASS_DICT)
        self.ironic_label_id = len(self.class_dict)
        self.class_dict[self.ironic_label_name] = self.ironic_label_id

        split = ds.get("data_split", {"a": 0})
        fold = split.get("fold", 0)
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

        self.video_transforms = {"train": transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),"val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),"test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}


    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        images = self._get_images(idx)

        aidx = int(self.audio_src_index[idx])
        audio = self._get_audio_from_index(aidx)
        face = self._get_face(idx)
        spec = self._get_spectrogram_from_index(aidx, audio)

        return {"data": {0: spec, 1: images, 2: audio, 3: face}, "label": int(self.label[idx]), "idx": idx}

    def _paths_from_id(self, uid: str) -> Tuple[str, str, str]:
        ap = os.path.join(self.audio_root, uid + ".wav")
        vp = os.path.join(self.visual_root, "Image-01-FPS", uid)
        fp = os.path.join(self.face_root, uid + ".npy")
        return ap, vp, fp

    def _valid_triplet(self, ap: str, vp: str, fp: str) -> bool:
        return os.path.exists(ap) and os.path.exists(vp) and os.path.exists(fp)

    def _split_inclusive(self, mode: str) -> None:
        self.norm_audio = {"total": {"mean": -7.1276217, "std": 5.116028}}

        train_rows = _read_csv_rows("./mydatasets/CREMAD/train.csv")
        test_rows = _read_csv_rows("./mydatasets/CREMAD/test.csv")

        train = self._collect_from_rows(train_rows)
        test = self._collect_from_rows(test_rows)

        total = self._merge_splits(train, test)
        X = np.array([total["item"], total["image"], total["audio"], total["faces"]]).T
        y = np.array(total["label"])

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=self.config.dataset.get("val_split_rate", 0.1),
            random_state=self.config.training_params.seed,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=self.config.dataset.get("val_split_rate", 0.1),
            random_state=self.config.training_params.seed,
            stratify=y_trainval,
        )

        if mode == "train":
            Xm, ym = X_train, y_train
        elif mode == "val":
            Xm, ym = X_val, y_val
        elif mode == "test":
            Xm, ym = X_test, y_test
        else:
            raise ValueError(f"mode must be one of train/val/test, got {mode!r}")

        self.item = Xm[:, 0].tolist()
        self.image = Xm[:, 1].tolist()
        self.audio = Xm[:, 2].tolist()
        self.faces = Xm[:, 3].tolist()
        self.label = ym.tolist()

    def _collect_from_rows(self, rows: List[List[str]]) -> Dict[str, List[Any]]:
        out = {"item": [], "image": [], "audio": [], "faces": [], "label": []}
        for uid, cls in rows:
            ap, vp, fp = self._paths_from_id(uid)
            if not self._valid_triplet(ap, vp, fp):
                continue
            out["item"].append(uid)
            out["image"].append(vp)
            out["audio"].append(ap)
            out["faces"].append(fp)
            out["label"].append(BASE_CLASS_DICT[cls])
        return out

    def _merge_splits(self, a: Dict[str, List[Any]], b: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return {k: a[k] + b[k] for k in a.keys()}

    def _split_noninclusive(self, fold: int, mode: str) -> None:
        with open("./mydatasets/CREMAD/normalization_audio.pkl", "r") as f:
            self.norm_audio = json.load(f)

        with open("./mydatasets/CREMAD/data_splits_VALV.pkl", "r") as f:
            splits = json.load(f)

        for entry in splits[str(fold + 1)][mode]:
            name = entry.split("-")[0]
            label = BASE_CLASS_DICT[name.split("_")[2]]
            ap = os.path.join(self.audio_root, name)
            vp = os.path.join(self.visual_root, f"Image-{self.fps:02d}-FPS", name.split(".")[0])
            fp = os.path.join(self.face_root, name.replace(".wav", ".npy"))
            if not self._valid_triplet(ap, vp, fp):
                continue
            self.item.append(name)
            self.image.append(vp)
            self.audio.append(ap)
            self.faces.append(fp)
            self.label.append(int(label))

    def _apply_ironic_pairs_if_enabled(self) -> None:
        if self.mode not in self.ironic_modes:
            return

        n = len(self.audio)
        if n < 2:
            return

        # rng = np.random.default_rng(int(self.config.training_params.seed))
        labels_np = np.asarray(self.label, dtype=np.int64)

        base_mask = labels_np != self.ironic_label_id
        base_labels = labels_np[base_mask]
        if base_labels.size == 0:
            return

        uniq, counts = np.unique(base_labels, return_counts=True)
        target = float(np.mean(counts))
        print("Ironic rate is {}".format(self.ironic_rate))
        k = int(round(float(self.ironic_rate) * target))
        k = max(0, min(k, int(base_mask.sum())))
        if k == 0:
            return

        idx_by_label: Dict[int, np.ndarray] = {}
        for lab in uniq.tolist():
            lab = int(lab)
            idx_by_label[lab] = np.where(labels_np == lab)[0]

        contradiction_map = {
            0: [],
            1: [2, 3, 4, 5],
            2: [1],
            3: [1],
            4: [1],
            5: [1],
        }

        candidates_for_flip = np.where(base_mask)[0]
        candidates_t = torch.as_tensor(candidates_for_flip, dtype=torch.long)
        candidates_t = torch.sort(candidates_t).values
        g = torch.Generator()
        g.manual_seed(int(self.config.training_params.seed))
        perm = candidates_t[torch.randperm(candidates_t.numel(), generator=g)]
        chosen = perm[:k].cpu().numpy()
        self.ironic_mask[chosen] = True

        per_src_count: Dict[int, int] = {int(l): 0 for l in uniq.tolist()}
        per_pair_count: Dict[Tuple[int, int], int] = {}
        used_donors: Dict[int, int] = {}

        for i in chosen:
            src_label = int(labels_np[i])
            preferred = contradiction_map.get(src_label, [])

            pool = []
            for lab in preferred:
                if lab in idx_by_label:
                    pool.append(idx_by_label[lab])

            if pool:
                cand = np.concatenate(pool)
                cand = cand[cand != i]
            else:
                cand = np.where(labels_np != src_label)[0]
                cand = cand[cand != i]

            if cand.size == 0:
                self.ironic_mask[i] = False
                continue

            cand_t = torch.as_tensor(cand, dtype=torch.long)
            cand_t = torch.sort(cand_t).values
            g = torch.Generator()
            g.manual_seed(int(self.config.training_params.seed+i))  # +i to decorrelate
            j = int(cand_t[torch.randint(0, cand_t.numel(), (1,), generator=g)].item())

            self.audio_src_index[i] = j
            self.label[i] = self.ironic_label_id

            donor_label = int(labels_np[j])
            per_src_count[src_label] = per_src_count.get(src_label, 0) + 1
            per_pair_count[(src_label, donor_label)] = per_pair_count.get((src_label, donor_label), 0) + 1
            used_donors[j] = used_donors.get(j, 0) + 1

        labels_after, counts_after = np.unique(np.asarray(self.label, dtype=np.int64), return_counts=True)
        dist_after = dict(zip(labels_after.tolist(), counts_after.tolist()))

        top_pairs = sorted(per_pair_count.items(), key=lambda x: x[1], reverse=True)[:10]
        top_donors = sorted(used_donors.items(), key=lambda x: x[1], reverse=True)[:10]

        self.logger.info(
            f"[{self.mode}] Ironic pairing enabled: ironic_rate={float(self.ironic_rate):.3f}, "
            f"target≈{target:.1f}, requested_k={k}, applied_k={int(self.ironic_mask.sum())}"
        )
        self.logger.info(f"[{self.mode}] Flipped-to-sarcasm counts by original class: {per_src_count}")
        self.logger.info(
            f"[{self.mode}] Top (orig_label -> donor_label) sarcasm pairs: "
            f"{[(f'{BASE_CLASS_DICT_INV[a]}->{BASE_CLASS_DICT_INV[b]}', c) for (a, b), c in top_pairs]}"
        )
        self.logger.info(f"[{self.mode}] Top donor indices reused (idx, times): {top_donors}")
        self.logger.info(f"[{self.mode}] Label distribution after applying ironic pairs: {dist_after}")

    def _load_or_build_wav_norm(self) -> None:
        path = self.config.dataset.get("norm_wav_path", None)
        norm = _safe_pickle_load(path)
        if norm is not None:
            self.wav_norm = norm
            self.logger.info(f"Loaded wav norm from {path}")
            return
        if self.mode != "train":
            return
        self.wav_norm = self._compute_wav_norm()
        save_to = path or "./mydatasets/CREMAD/wav_norm.pkl"
        self.logger.warning(f"Saving wav norm to {save_to}")
        _safe_pickle_save(save_to, self.wav_norm)

    def _compute_wav_norm(self) -> Dict[str, Any]:
        count = 0
        wav_sum = 0.0
        wav_sqsum = 0.0
        max_duration = 0.0

        for cur_wav in tqdm(self.audio):
            audio, sr = torchaudio.load(cur_wav)
            audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)[0]
            max_duration = max(max_duration, float(audio.shape[0]) / float(self.sampling_rate))
            wav_sum += float(torch.sum(audio))
            wav_sqsum += float(torch.sum(audio**2))
            count += int(audio.numel())

        mean = wav_sum / count
        var = (wav_sqsum / count) - (mean**2)
        std = float(np.sqrt(var))
        return {"mean": mean, "std": std, "max_duration": max_duration}

    def _load_or_build_face_norm(self) -> None:
        path = self.config.dataset.get("norm_face_path", None)
        norm = _safe_pickle_load(path)
        if norm is not None:
            self.face_norm = norm
            self.logger.info(f"Loaded face norm from {path}")
            return
        if self.mode != "train":
            return
        self.face_norm = self._compute_face_norm()
        save_to = path or "face_norm.pkl"
        self.logger.warning(f"Saving face norm to {save_to}")
        _safe_pickle_save(save_to, self.face_norm)

    def _compute_face_norm(self) -> Dict[str, Any]:
        count = 0
        s = 0.0
        ss = 0.0
        max_faces = 0

        for cur in tqdm(self.faces):
            feats = np.load(cur, allow_pickle=True)
            s += np.sum(feats, axis=0)
            ss += np.sum(feats**2, axis=0)
            count += feats.shape[0]
            max_faces = max(max_faces, feats.shape[0])

        mean = s / count
        var = (ss / count) - (mean**2)
        std = np.sqrt(var)
        return {"mean": mean, "std": std, "max_faces": int(max_faces)}

    def _get_images(self, idx: int):
        if not self.return_data.get("video", False):
            return False
        t = self.video_transforms[self.mode]
        frames = sorted(os.listdir(str(self.image[idx])))
        images = torch.zeros((self.num_frame, 3, 224, 224), dtype=torch.float32)
        for i, fn in enumerate(frames[: self.num_frame]):
            img = Image.open(os.path.join(self.image[idx], fn)).convert("RGB")
            images[i] = t(img)
        return torch.permute(images, (1, 0, 2, 3))

    def _get_audio_from_index(self, aidx: int):
        if not self.return_data.get("audio", False):
            return False
        audio, sr = torchaudio.load(self.audio[aidx])
        audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)[0]
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

    def _get_spectrogram_from_index(self, aidx: int, audio):
        if not self.return_data.get("spectrogram", False):
            return False

        if audio is False:
            samples, _ = librosa.load(self.audio[aidx], sr=self.sampling_rate)
            resamples = np.tile(samples, 3)[: self.sampling_rate * 3]
            resamples = np.clip(resamples, -1.0, 1.0)
        else:
            a = audio.detach().cpu().numpy()
            resamples = np.tile(a, 3)[: self.sampling_rate * 3]
            resamples = np.clip(resamples, -1.0, 1.0)

        spec = librosa.stft(resamples, n_fft=512, hop_length=353)
        spec = np.log(np.abs(spec) + 1e-7)

        if self.norm_type == "per_sample":
            m, s = float(np.mean(spec)), float(np.std(spec))
            spec = (spec - m) / (s + 1e-9)
        elif self.norm_type == "per_freq":
            m = np.array(self.norm_audio["per_req"]["mean"])
            s = np.array(self.norm_audio["per_req"]["std"])
            spec = ((spec.T - m) / (s + 1e-9)).T
        elif self.norm_type == "total":
            m = float(self.norm_audio["total"]["mean"])
            s = float(self.norm_audio["total"]["std"])
            spec = (spec - m) / (s + 1e-9)

        return torch.from_numpy(spec)

class CramedDPlus_Dataloader:
    def __init__(self, config: Any):
        self.config = config
        train_ds, val_ds, test_ds = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(int(self.config.training_params.seed))

        # num_workers = int(getattr(self.config.training_params, "data_loader_workers", 0) or 0)
        num_workers = int(0)

        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.config.training_params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.training_params.pin_memory,
            generator=g,
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
        train_dataset = CremadPlusDataset(config=self.config, mode="train")
        valid_dataset = CremadPlusDataset(config=self.config, mode="val")
        test_dataset = CremadPlusDataset(config=self.config, mode="test")
        return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    """
    Determinism smoke-test:
    - Build the dataloader twice in the same process
    - Check that dataset bookkeeping (labels, ironic_mask, audio_src_index, paths) matches exactly
    - Check that first batch (and optionally a few batches) match exactly

    Notes:
    - This assumes your transforms / audio pipeline are deterministic under the same seeds.
    - If you use num_workers>0, you MUST seed torch in _seed_worker (see below).
    - If you keep RandomResizedCrop/Flip in train mode, this will still be deterministic
      as long as torch RNG is seeded consistently.
    """

    import os
    import random
    import numpy as np
    import torch
    from easydict import EasyDict as ED

    # -------------------------
    # Optional: fix _seed_worker
    # -------------------------
    # Make sure your _seed_worker includes torch.manual_seed(worker_seed)
    # def _seed_worker(worker_id: int) -> None:
    #     worker_seed = torch.initial_seed() % 2**32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #     torch.manual_seed(worker_seed)

    def to_ed(x):
        if isinstance(x, dict):
            return ED({k: to_ed(v) for k, v in x.items()})
        if isinstance(x, list):
            return [to_ed(v) for v in x]
        return x

    cfg_json = {
        "dataset": {
            "dataloader_class": "CramedDPlus_Dataloader",
            "data_roots": "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CremaD/CREMA-D",
            "norm_wav_path": "wav_norm_22050_fold{}.pkl",
            "norm_face_path": "norm_face_path_fold{}.pkl",
            "return_data": {"video": True, "spectrogram": True, "audio": False, "face": False},
            "data_split": {"method": "non_inclusive", "fold": 0},
            "fps": 1,
            "sampling_rate": 22050,
            "num_frame": 3,
            "ironic_rate": 0.2,
            "ironic_label_name": "SARCASM",
            "ironic_modes": ["train"],
        },
        "training_params": {
            "seed": 0,
            "batch_size": 10,
            "test_batch_size": 10,
            "pin_memory": False,
            "data_loader_workers": 0,  # keep 0 for simplest determinism check
        },
    }
    config = to_ed(cfg_json)

    def seed_everything(seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism on GPU ops (may slow down):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def compare_lists(name, a, b):
        if a != b:
            # show first mismatch
            for i, (xa, xb) in enumerate(zip(a, b)):
                if xa != xb:
                    raise AssertionError(f"{name} differs at idx={i}: {xa!r} != {xb!r}")
            raise AssertionError(f"{name} differs (length?) {len(a)} != {len(b)}")

    def compare_numpy(name, a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape:
            raise AssertionError(f"{name} shape differs: {a.shape} != {b.shape}")
        if a.dtype != b.dtype:
            raise AssertionError(f"{name} dtype differs: {a.dtype} != {b.dtype}")
        if not np.array_equal(a, b):
            # find first mismatch
            idx = np.flatnonzero(a != b)[0]
            raise AssertionError(f"{name} differs at flat idx={idx}: {a.flat[idx]} != {b.flat[idx]}")

    def compare_tensors(name, a: torch.Tensor, b: torch.Tensor):
        if a.shape != b.shape:
            raise AssertionError(f"{name} shape differs: {tuple(a.shape)} != {tuple(b.shape)}")
        if a.dtype != b.dtype:
            raise AssertionError(f"{name} dtype differs: {a.dtype} != {b.dtype}")
        if not torch.equal(a, b):
            # locate first mismatch (works best on CPU)
            aa = a.detach().cpu()
            bb = b.detach().cpu()
            diff = (aa != bb).flatten()
            where = torch.nonzero(diff, as_tuple=False)
            if where.numel() > 0:
                j = int(where[0].item())
                raise AssertionError(f"{name} differs at flat idx={j}: {aa.flatten()[j]} != {bb.flatten()[j]}")
            raise AssertionError(f"{name} differs (unknown location)")

    def build_loaders_with_seed(seed: int):
        # important: reset global seeds BEFORE constructing datasets/loaders
        seed_everything(seed)
        # IMPORTANT: your dataloader currently hardcodes generator seed to 0.
        # If you updated it to config.training_params.seed, even better.
        return CramedDPlus_Dataloader(config)

    def check_dataset_bookkeeping(ds1: CremadPlusDataset, ds2: CremadPlusDataset):
        assert len(ds1) == len(ds2), f"len differs: {len(ds1)} != {len(ds2)}"

        compare_lists("item", ds1.item, ds2.item)
        compare_lists("image", ds1.image, ds2.image)
        compare_lists("audio", ds1.audio, ds2.audio)
        compare_lists("faces", ds1.faces, ds2.faces)
        compare_lists("label(list)", [int(x) for x in ds1.label], [int(x) for x in ds2.label])

        compare_numpy("audio_src_index", np.asarray(ds1.audio_src_index), np.asarray(ds2.audio_src_index))
        compare_numpy("ironic_mask", np.asarray(ds1.ironic_mask), np.asarray(ds2.ironic_mask))

        # also check the class dict & ironic label id
        assert ds1.class_dict == ds2.class_dict, "class_dict differs"
        assert int(ds1.ironic_label_id) == int(ds2.ironic_label_id), "ironic_label_id differs"

    def check_first_k_items(ds1: CremadPlusDataset, ds2: CremadPlusDataset, k: int = 8):
        # Ensure deterministic per-sample generation by resetting seeds before each access
        # (esp. because train mode transform uses randomness)
        for idx in range(min(k, len(ds1))):
            seed_everything(int(config.training_params.seed))

            a = ds1[idx]
            seed_everything(int(config.training_params.seed))
            b = ds2[idx]

            # Compare label and index
            assert int(a["label"]) == int(b["label"]), f"label differs at idx={idx}"
            assert int(a["idx"]) == int(b["idx"]), f"idx differs at idx={idx}"

            # Compare data tensors if present
            da, db = a["data"], b["data"]

            # spec
            if da.get(0, False) is not False and db.get(0, False) is not False:
                compare_tensors(f"spec(idx={idx})", da[0], db[0])

            # video
            if da.get(1, False) is not False and db.get(1, False) is not False:
                compare_tensors(f"video(idx={idx})", da[1], db[1])

            # audio
            if da.get(2, False) is not False and db.get(2, False) is not False:
                compare_tensors(f"audio(idx={idx})", da[2], db[2])

            # face
            if da.get(3, False) is not False and db.get(3, False) is not False:
                compare_tensors(f"face(idx={idx})", da[3], db[3])

    def check_first_n_batches(loader1, loader2, n: int = 2):
        it1 = iter(loader1)
        it2 = iter(loader2)
        for bi in range(n):
            b1 = next(it1)
            b2 = next(it2)

            compare_tensors(f"batch[{bi}].label", b1["label"], b2["label"])
            compare_tensors(f"batch[{bi}].idx", b1["idx"], b2["idx"])

            # Compare data entries if present
            for key in [0, 1, 2, 3]:
                if key in b1["data"] or key in b2["data"]:
                    assert key in b1["data"] and key in b2["data"], f"batch[{bi}].data missing key {key} in one loader"
                    compare_tensors(f"batch[{bi}].data[{key}]", b1["data"][key], b2["data"][key])

            # Compare attention masks if present
            for key in ["attention_mask_audio", "attention_mask_face"]:
                if key in b1["data"] or key in b2["data"]:
                    assert key in b1["data"] and key in b2["data"], f"batch[{bi}].data missing {key} in one loader"
                    compare_tensors(f"batch[{bi}].data[{key}]", b1["data"][key], b2["data"][key])

    # -------------------------
    # Run the checks
    # -------------------------
    SEED = int(config.training_params.seed)

    loaders_a = build_loaders_with_seed(SEED)
    loaders_b = build_loaders_with_seed(SEED)

    ds_a = loaders_a.train_loader.dataset
    ds_b = loaders_b.train_loader.dataset

    print("Checking dataset bookkeeping...")
    check_dataset_bookkeeping(ds_a, ds_b)
    print("✓ Dataset bookkeeping identical")

    print("Checking first few __getitem__ outputs...")
    check_first_k_items(ds_a, ds_b, k=8)
    print("✓ First few items identical")

    print("Checking first few DataLoader batches...")
    check_first_n_batches(loaders_a.train_loader, loaders_b.train_loader, n=2)
    print("✓ First few batches identical")

    print("\nALL DETERMINISM CHECKS PASSED ✅")
