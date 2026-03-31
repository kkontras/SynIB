#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONDOR_DIR = REPO_ROOT / "run" / "condor"
LOGS_DIR = CONDOR_DIR / "logs"

STRICT_REQUIREMENTS = (
    '(GPUs_GlobalMemoryMb > 8000) &&'
    '(machine != "spinel.esat.kuleuven.be") && '
    '(machine != "raidho.esat.kuleuven.be") && '
    '(machine != "thurisaz.esat.kuleuven.be") && '
    '(machine != "spchcl18.esat.kuleuven.be") && '
    '(machine != "psi-test2.esat.kuleuven.be") && '
    '(machine != "wulfenite.esat.kuleuven.be") && '
    '(machine != "vladimir.esat.kuleuven.be") && '
    '(machine != "estragon.esat.kuleuven.be") && '
    '(machine != "daisen.esat.kuleuven.be") && '
    '(machine != "fuji.esat.kuleuven.be") && '
    '(machine != "goryu.esat.kuleuven.be") && '
    '(machine != "iwaki.esat.kuleuven.be") && '
    '(machine != "jonen.esat.kuleuven.be")'
)

FOLD_RE = re.compile(r"fold(?:fold)?(\d+)")
L_RE = re.compile(r"_l([0-9.]+)_vldaccuracy")
LR_RE = re.compile(r"_lr([0-9.]+)_wd")
WD_RE = re.compile(r"_wd([0-9.]+)(?:_|\.pth)")
LSPARSE_RE = re.compile(r"_lsparse([0-9.]+)")
PMIN_RE = re.compile(r"_pmin([0-9.]+)")
LOG_PROC_RE = re.compile(r"^\d+_(\d+)\.log$")

DATASETS = {
    "mosi": {
        "default_config": "run/configs/FactorCL/Mosi/default_config_mosi_VT.json",
        "dir": "run/configs/FactorCL/Mosi/syn/VT",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/Balance_Final/MOSI/VT"),
    },
    "mosei": {
        "default_config": "run/configs/FactorCL/Mosei/default_config_mosei_VT_syn.json",
        "dir": "run/configs/FactorCL/Mosei/syn/VT",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Rmask/MOSEI/VT"),
    },
    "mustard": {
        "default_config": "run/configs/FactorCL/Mustard/default_config_mustard_VT.json",
        "dir": "run/configs/FactorCL/Mustard/release/VT",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/MUSTARD"),
    },
    "urfunny": {
        "default_config": "run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json",
        "dir": "run/configs/FactorCL/URFunny/release/VT",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/URFUNNY"),
    },
}

VARIANTS = {
    "synib": {
        "config_name": {
            "mosi": "synprom_RMask.json",
            "mosei": "synprom_RMask.json",
            "mustard": "synprom_RMask.json",
            "urfunny": "synprom_RMask.json",
        },
        "grid": "legacy",
        "use_existing_args": False,
        "log_dir_suffix": "synib_recover",
    },
    "synib_nonpre": {
        "config_name": {
            "mosi": "synprom_RMask_nopre.json",
            "mosei": "synprom_RMask_nopre.json",
            "mustard": "synprom_RMask_nonpre.json",
            "urfunny": "synprom_RMask_nopre.json",
        },
        "grid": "legacy_nonpre",
        "use_existing_args": False,
        "log_dir_suffix": "synib_nonpre_recover",
    },
    "synibu": {
        "existing_args": {
            "mosi": CONDOR_DIR / "mosi_synibu.args",
            "mosei": CONDOR_DIR / "mosei_synibu.args",
            "mustard": CONDOR_DIR / "mustard_synibu.args",
            "urfunny": CONDOR_DIR / "urfunny_synibu.args",
        },
        "use_existing_args": True,
        "log_source": {
            "mosi": LOGS_DIR / "mosi_synibu",
            "mosei": LOGS_DIR / "mosei_synibu",
            "mustard": LOGS_DIR / "mustard_synibu",
            "urfunny": LOGS_DIR / "urfunny_synibu",
        },
        "log_dir_suffix": "synibu_recover",
    },
    "synibu_nonpre": {
        "existing_args": {
            "mosi": CONDOR_DIR / "mosi_synibu_nonpre.args",
            "mosei": CONDOR_DIR / "mosei_synibu_nonpre.args",
            "mustard": CONDOR_DIR / "mustard_synibu_nonpre.args",
            "urfunny": CONDOR_DIR / "urfunny_synibu_nonpre.args",
        },
        "use_existing_args": True,
        "log_source": {
            "mosi": LOGS_DIR / "mosi_synibu_nonpre",
            "mosei": LOGS_DIR / "mosei_synibu_nonpre",
            "mustard": LOGS_DIR / "mustard_synibu_nonpre",
            "urfunny": LOGS_DIR / "urfunny_synibu_nonpre",
        },
        "log_dir_suffix": "synibu_nonpre_recover",
    },
}


def detect_mask(name: str) -> str:
    lowered = name.lower()
    if "perturblearned" in lowered or "perturnlearned" in lowered:
        return "learned"
    if "perturbrandom" in lowered or "perturnrandom" in lowered:
        return "random"
    return "none"


def parse_name_fields(name: str) -> tuple[str, str | None, str | None, str | None, str | None, str | None, int | None]:
    fold_match = FOLD_RE.search(name)
    l = L_RE.search(name)
    lr = LR_RE.search(name)
    wd = WD_RE.search(name)
    lsparse = LSPARSE_RE.search(name)
    pmin = PMIN_RE.search(name)
    return (
        detect_mask(name),
        l.group(1) if l else None,
        lsparse.group(1) if lsparse else None,
        pmin.group(1) if pmin else None,
        lr.group(1) if lr else None,
        wd.group(1) if wd else None,
        int(fold_match.group(1)) if fold_match else None,
    )


def variant_from_checkpoint_name(name: str) -> str | None:
    if "SynIB_RMask_U" in name:
        return None
    if "SynIB_RMask_nonpre" in name or "SynIB_RMask_nopre" in name:
        return "synib_nonpre"
    if "SynIB_RMask" in name:
        return "synib"
    return None


def expected_rows(variant: str, dataset: str) -> list[tuple[str, str | None, str | None, str | None, str, str, int]]:
    rows: list[tuple[str, str | None, str | None, str | None, str, str, int]] = []
    folds = [0, 1, 2]
    if variant in {"synibu", "synibu_nonpre"}:
        learned_l = ["0.001", "0.01", "0.1", "1", "10", "100"]
        learned_lsparse = ["0.001", "0.01", "0.1", "1", "10", "100"]
        random_l = learned_l
        random_pmin = ["0.3", "0.5", "0.8"]
        lr, wd = "0.0005", "0.001"
        for fold in folds:
            for l in learned_l:
                for lsparse in learned_lsparse:
                    rows.append(("learned", l, lsparse, None, lr, wd, fold))
            for l in random_l:
                for pmin in random_pmin:
                    rows.append(("random", l, None, pmin, lr, wd, fold))
        return rows

    if variant == "synib":
        learned_l = ["0.001", "0.01", "0.1", "1"]
        learned_lsparse = ["0.001", "0.01", "0.1", "1", "3", "5", "10"]
        random_l = ["0.001", "0.01", "0.1", "1"]
        random_pmin = ["0.1", "0.3", "0.5", "0.7", "0.9"]
        lr, wd = ("0.001", "0.001") if dataset == "urfunny" else ("0.0005", "0.001")
        for fold in folds:
            for l in learned_l:
                for lsparse in learned_lsparse:
                    rows.append(("learned", l, lsparse, None, lr, wd, fold))
            for l in random_l:
                for pmin in random_pmin:
                    rows.append(("random", l, None, pmin, lr, wd, fold))
        return rows

    if variant == "synib_nonpre":
        if dataset in {"mustard", "urfunny"}:
            return rows
        learned_l = ["0.001", "0.01", "0.1", "1"]
        learned_lsparse = ["0.001", "0.01", "0.1", "1", "3", "5", "10"]
        random_l = ["0.001", "0.01", "0.1", "1"]
        random_pmin = ["0.1", "0.3", "0.5", "0.7", "0.9"]
        lr, wd = "0.0005", "0.001"
        for fold in folds:
            for l in learned_l:
                for lsparse in learned_lsparse:
                    rows.append(("learned", l, lsparse, None, lr, wd, fold))
            for l in random_l:
                for pmin in random_pmin:
                    rows.append(("random", l, None, pmin, lr, wd, fold))
    return rows


def checkpoint_presence(dataset: str) -> defaultdict[str, set[tuple[str, str | None, str | None, str | None, str | None, str | None, int]]]:
    root = DATASETS[dataset]["checkpoint_root"]
    out: defaultdict[str, set[tuple[str, str | None, str | None, str | None, str | None, str | None, int]]] = defaultdict(set)
    if not root.exists():
        return out
    for path in root.glob("SynIB_RMask*.pth.tar"):
        variant = variant_from_checkpoint_name(path.name)
        if variant is None:
            continue
        fields = parse_name_fields(path.name)
        if fields[-1] is None:
            continue
        out[variant].add(fields)
    return out


def proc_status_map(log_dir: Path) -> dict[int, str]:
    status: dict[int, str] = {}
    if not log_dir.exists():
        return status
    for path in log_dir.glob("*.log"):
        match = LOG_PROC_RE.match(path.name)
        if not match:
            continue
        proc = int(match.group(1))
        text = path.read_text(errors="ignore")
        if "005 (" in text:
            if "Normal termination (return value 0)" in text:
                status[proc] = "done"
            elif status.get(proc) != "done":
                status[proc] = "fail"
        elif status.get(proc) != "done":
            status[proc] = "active"
    return status


def render_arg_line(dataset: str, variant: str, row: tuple[str, str | None, str | None, str | None, str, str, int]) -> str:
    mask, l, lsparse, pmin, lr, wd, fold = row
    base = DATASETS[dataset]
    cfg_name = VARIANTS[variant]["config_name"][dataset]
    config = f"{base['dir']}/{cfg_name}"
    parts = [
        "--config", config,
        "--default_config", base["default_config"],
        "--validate_with", "accuracy",
        "--batch_size", "32",
        "--no_model_save",
        "--lr", lr,
        "--wd", wd,
        "--fold", str(fold),
        "--rmask", mask,
        "--l", l or "0",
        "--start_over",
    ]
    if mask == "learned" and lsparse is not None:
        parts.extend(["--lsparse", lsparse])
    if mask == "random" and pmin is not None:
        parts.extend(["--pmin", pmin])
    return " ".join(parts)


def write_job(job_path: Path, args_path: Path, log_dir: Path) -> None:
    content = "\n".join([
        "Notification = Error",
        f"executable   = {REPO_ROOT / 'run/condor/run_train.sh'}",
        f"initialdir   = {log_dir}",
        "Log          = $(ClusterId)_$(Process).log",
        "Output       = $(ClusterId)_$(Process).out",
        "Error        = $(ClusterId)_$(Process).err",
        "RequestCpus  = 8",
        "RequestMemory = 32G",
        "RequestWalltime = 100000",
        "Request_GPUs = 1",
        "NiceUser     = True",
        f"Requirements = {STRICT_REQUIREMENTS}",
        f"Queue Arguments from {args_path}",
        "",
    ])
    job_path.write_text(content)


def main() -> None:
    created: list[tuple[str, Path, int]] = []
    checkpoint_map = {ds: checkpoint_presence(ds) for ds in DATASETS}

    for dataset in DATASETS:
        for variant in VARIANTS:
            recover_args = CONDOR_DIR / f"{dataset}_{variant}_recover.args"
            recover_job = CONDOR_DIR / f"{dataset}_{variant}_recover.job"
            log_dir = LOGS_DIR / f"{dataset}_{VARIANTS[variant]['log_dir_suffix']}"
            log_dir.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []

            if VARIANTS[variant]["use_existing_args"]:
                args_lines = VARIANTS[variant]["existing_args"][dataset].read_text().splitlines()
                statuses = proc_status_map(VARIANTS[variant]["log_source"][dataset])
                for proc, line in enumerate(args_lines):
                    state = statuses.get(proc)
                    if state == "done" or state == "active":
                        continue
                    if line.strip():
                        lines.append(line.strip())
            else:
                found = checkpoint_map[dataset].get(variant, set())
                for row in expected_rows(variant, dataset):
                    if row not in found:
                        lines.append(render_arg_line(dataset, variant, row))

            recover_args.write_text("\n".join(lines) + ("\n" if lines else ""))
            write_job(recover_job, recover_args, log_dir)
            created.append((f"{dataset}_{variant}", recover_job, len(lines)))

    for name, job, count in created:
        print(f"{name}\t{count}\t{job}")


if __name__ == "__main__":
    main()
