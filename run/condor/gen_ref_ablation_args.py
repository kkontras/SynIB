#!/usr/bin/env python3
"""Generate the condor args list for the rebuttal reference-distribution ablation.

Core matrix (45 runs): 3 references x 3 folds/seeds on
UR-Funny, MUStARD, MOSI, Hateful Memes, CREMA-D-Irony(alpha=0.5).
Optional waves (emitted to separate files):
  - cremad alpha=0.1 (3 refs x 3 folds)
  - lambda mini-sweep {0.5, 2}xlambda* for uniform/unimodal_anchor on MUStARD + CREMA-D, fold 0.

Every command replicates the paper's M_random command from docs/REPRODUCE.md exactly
(same lambda, pi, lr, wd, batch size), adding only --reference_type/--ref_diag/--tag.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = "run/configs/rebuttal_ref_ablation"
REFS = ["uniform", "class_prior", "unimodal_anchor"]
FOLDS = [0, 1, 2]

# dataset -> (config, default_config, paper M_random flags)
DATASETS = {
    "urfunny": (f"{CFG}/urfunny_synib.json", f"{CFG}/urfunny_default.json",
                "--rmask random --l 0.001 --perturb_pmin 0.7 --perturb_fill ema --lr 0.001 --wd 0.001"),
    "mustard": (f"{CFG}/mustard_synib.json", f"{CFG}/mustard_default.json",
                "--rmask random --l 0.001 --perturb_pmin 0.1 --perturb_fill ema --lr 0.0005 --wd 0.001"),
    "mosi":    (f"{CFG}/mosi_synib_u.json", f"{CFG}/mosi_default.json",
                "--rmask random --l 0.1 --perturb_pmin 0.3 --perturb_fill ema --lr 0.0005 --wd 0.001 --batch_size 32"),
    "hm":      (f"{CFG}/hm_synib_u_merged.json", f"{CFG}/hm_default_config.json",
                "--rmask random --l 0.01 --l_pareto 0.1 --perturb_pmin 0.3 --perturb_pmax 0.5"),
    "cremad":  (f"{CFG}/cremad_synib_u.json", f"{CFG}/cremad_default.json",
                "--ironic_rate 0.5 --rmask random --l 1.0 --perturb_pmin 0.20 --perturb_fill ema"),
}

LAMBDA_STAR = {"mustard": 0.001, "cremad": 1.0}


def line(cfg, dflt, flags, fold, ref, tag):
    return (f"--config {cfg} --default_config {dflt} --fold {fold} {flags} "
            f"--reference_type {ref} --ref_diag --start_over --tag {tag}")


def main():
    core, alpha01, lam = [], [], []
    # priority order per the plan: cremad > mustard > hm > mosi > urfunny; anchor/uniform before class_prior
    ds_order = ["cremad", "mustard", "hm", "mosi", "urfunny"]
    ref_order = ["uniform", "unimodal_anchor", "class_prior"]
    for ref in ref_order:
        for ds in ds_order:
            cfg, dflt, flags = DATASETS[ds]
            for fold in FOLDS:
                core.append(line(cfg, dflt, flags, fold, ref, f"REFABL_{ds}"))

    cfg, dflt, flags = DATASETS["cremad"]
    flags01 = flags.replace("--ironic_rate 0.5", "--ironic_rate 0.1")
    for ref in ref_order:
        for fold in FOLDS:
            alpha01.append(line(cfg, dflt, flags01, fold, ref, "REFABL_cremad_a01"))

    for ds in ("mustard", "cremad"):
        cfg, dflt, flags = DATASETS[ds]
        for ref in ("uniform", "unimodal_anchor"):
            for mult, mtag in ((0.5, "lam0.5x"), (2.0, "lam2x")):
                lam_val = LAMBDA_STAR[ds] * mult
                f2 = flags.replace(f"--l {LAMBDA_STAR[ds]}", f"--l {lam_val}")
                lam.append(line(cfg, dflt, f2, 0, ref, f"REFABL_{ds}_{mtag}"))

    for name, rows in (("ref_ablation_core.args", core),
                       ("ref_ablation_alpha01.args", alpha01),
                       ("ref_ablation_lambda.args", lam)):
        p = os.path.join(HERE, name)
        with open(p, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{p}: {len(rows)} jobs")


if __name__ == "__main__":
    main()
