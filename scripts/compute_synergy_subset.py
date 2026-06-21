"""Compute synergy-subset stats from existing unimodal CEU test pickles.

Synergy subset := test examples misclassified by ALL unimodal predictors
(strict: every seed × every modality wrong).
Also reports the looser per-modality-majority criterion for comparison.
"""
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CEU = ROOT / "artifacts" / "ceus"

# (dataset_name, ceu_pickle, modality_for_each_fold)
# Folds 0..5 in each pickle are 3 seeds × 2 modalities.
# For Factor-CL pickles fold 0..2 = video, 3..5 = text (per checkpoint paths).
# For HM pickle fold 0..2 = text, 3..5 = image (per checkpoint paths).
SOURCES = [
    ("MOSI",    CEU / "mosi/mosi_ceu_test.pkl",
     {0: "video", 1: "video", 2: "video", 3: "text", 4: "text", 5: "text"}),
    ("MUSTARD", CEU / "mustard/mustard_ceu_test.pkl",
     {0: "video", 1: "video", 2: "video", 3: "text", 4: "text", 5: "text"}),
    ("UR-FUNNY", CEU / "ur_funny/ur_funny_ceu_test.pkl",
     {0: "video", 1: "video", 2: "video", 3: "text", 4: "text", 5: "text"}),
    ("HatefulMemes", CEU / "hatefulmemes/hatefulmemes_ceu_test_small_tf_deberta.pkl",
     {0: "text", 1: "text", 2: "text", 3: "image", 4: "image", 5: "image"}),
]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze(name, pkl_path, modality_map):
    d = load(pkl_path)
    folds = d["folds"]
    # Sanity: all folds same length, same targets.
    target_ref = np.array(folds[0]["targets"])
    N = target_ref.shape[0]
    for fk, fv in folds.items():
        t = np.array(fv["targets"])
        assert t.shape[0] == N, f"{name} fold {fk}: size {t.shape[0]} vs {N}"
        assert np.array_equal(t, target_ref), f"{name} fold {fk}: target mismatch"

    # Per-fold correctness mask.
    correct = {}
    for fk, fv in folds.items():
        p = np.array(fv["preds_combined"])
        yhat = p.argmax(-1) if p.ndim == 2 else p
        correct[fk] = (yhat == target_ref)

    by_mod = {}  # modality -> list of correctness masks
    for fk, mod in modality_map.items():
        by_mod.setdefault(mod, []).append(correct[fk])

    print(f"\n=== {name}  (N={N})  source: {pkl_path.relative_to(ROOT)}")

    # Per-seed unimodal accuracies.
    for mod, masks in by_mod.items():
        accs = [m.mean() for m in masks]
        print(f"  {mod} seeds: " + ", ".join(f"{a:.4f}" for a in accs)
              + f"  (mean={np.mean(accs):.4f})")

    # Strict criterion: misclassified by ALL 6 unimodal predictors.
    all_wrong_strict = np.ones(N, dtype=bool)
    for mod, masks in by_mod.items():
        for m in masks:
            all_wrong_strict &= ~m
    n_strict = int(all_wrong_strict.sum())

    # Looser: per modality, majority vote across seeds; example wrong by a
    # modality iff at least 2/3 seeds wrong; synergy = all modalities wrong.
    all_wrong_majority = np.ones(N, dtype=bool)
    for mod, masks in by_mod.items():
        wrong_count = np.zeros(N, dtype=int)
        for m in masks:
            wrong_count += (~m).astype(int)
        mod_wrong_majority = wrong_count >= (len(masks) // 2 + 1)
        all_wrong_majority &= mod_wrong_majority
    n_majority = int(all_wrong_majority.sum())

    print(f"  synergy subset (STRICT, all 6 unimodals wrong):    "
          f"n={n_strict}/{N} = {100*n_strict/N:.2f}%")
    print(f"  synergy subset (MAJORITY, both mods majority-wrong): "
          f"n={n_majority}/{N} = {100*n_majority/N:.2f}%")

    return {
        "dataset": name,
        "N": N,
        "subset_strict": n_strict,
        "subset_majority": n_majority,
        "strict_mask": all_wrong_strict,
        "majority_mask": all_wrong_majority,
        "targets": target_ref,
    }


def main():
    results = []
    for name, path, modmap in SOURCES:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        results.append(analyze(name, path, modmap))

    print("\n=== Summary table")
    print(f"{'dataset':<14} {'N':>5} {'strict':>8} {'strict%':>8} "
          f"{'majority':>10} {'maj%':>7}")
    for r in results:
        print(f"{r['dataset']:<14} {r['N']:>5} {r['subset_strict']:>8} "
              f"{100*r['subset_strict']/r['N']:>7.2f}% "
              f"{r['subset_majority']:>10} "
              f"{100*r['subset_majority']/r['N']:>6.2f}%")

    # Persist masks for downstream per-method evaluation.
    out = ROOT / "artifacts" / "synergy_subset_masks.pkl"
    payload = {
        r["dataset"]: {
            "N": r["N"],
            "targets": r["targets"],
            "strict_mask": r["strict_mask"],
            "majority_mask": r["majority_mask"],
        } for r in results
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nWrote masks → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
