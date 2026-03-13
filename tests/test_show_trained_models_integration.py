import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synib.entrypoints import show as show_entrypoint


SHOW_SUITES = {
    "cremad_irony": {
        "default_config": ROOT_DIR / "run/configs/CREMA_D/default_config_cremadplus_res_syn.json",
        "checkpoint_dir": Path(
            "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/CremadPlus/v2"
        ),
    },
    "factorcl_mosi_vt": {
        "default_config": ROOT_DIR / "run/configs/FactorCL/Mosi/default_config_mosi_VT.json",
        "checkpoint_dir": Path(
            "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/Balance_Final/MOSI/VT"
        ),
    },
    "factorcl_urfunny_vt": {
        "default_config": ROOT_DIR / "run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json",
        "checkpoint_dir": Path(
            "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/URFUNNY"
        ),
    },
    "factorcl_mustard_vt": {
        "default_config": ROOT_DIR / "run/configs/FactorCL/Mustard/default_config_mustard_VT.json",
        "checkpoint_dir": Path(
            "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/MUSTARD"
        ),
    },
}


def _selected_suite_names():
    raw = os.getenv("SYNIB_SHOW_INTEGRATION_DATASETS", "").strip()
    if not raw:
        return list(SHOW_SUITES.keys())
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    return [name for name in requested if name in SHOW_SUITES]


def _max_models_per_suite():
    raw = os.getenv("SYNIB_SHOW_INTEGRATION_MAX_MODELS", "").strip()
    if not raw:
        return 0
    return max(int(raw), 0)


class _Args(dict):
    def __getattr__(self, item):
        return self[item]


def _show_args():
    return _Args(
        config=None,
        default_config=None,
        fold=None,
        alpha=None,
        validate_with=None,
        tanh_mode_beta=None,
        regby=None,
        batch_size=None,
        l=None,
        multil=None,
        lib=None,
        kmepoch=None,
        num_samples=None,
        contrcoeff=None,
        contr_type=None,
        shuffle_type=None,
        num_classes=None,
        optim_method=None,
        ending_epoch=None,
        load_ongoing=None,
        recon_weight1=None,
        recon_weight2=None,
        recon_epochstages=None,
        recon_ensemblestages=None,
        lr=None,
        wd=None,
        cls=None,
        printing=False,
        ironic_rate=None,
        perturb=None,
        perturb_fill=None,
        perturb_pmax=None,
        perturb_pmin=None,
        perturb_lsparse=None,
        rmask=None,
        pmin=None,
        pmax=None,
        lsparse=None,
        pre=False,
        frozen=False,
        tdqm_disable=False,
        start_over=False,
    )


class ShowTrainedModelsIntegrationTest(unittest.TestCase):
    def test_show_can_load_existing_trained_checkpoints(self):
        max_models = _max_models_per_suite()
        suites_run = 0

        for suite_name in _selected_suite_names():
            suite = SHOW_SUITES[suite_name]
            checkpoint_dir = suite["checkpoint_dir"]
            if not checkpoint_dir.exists():
                continue

            checkpoints = sorted(checkpoint_dir.glob("*.pth.tar"))
            if max_models:
                checkpoints = checkpoints[:max_models]
            if not checkpoints:
                continue

            suites_run += 1
            for checkpoint_path in checkpoints:
                with self.subTest(suite=suite_name, checkpoint=checkpoint_path.name):
                    override = {
                        "model": {
                            "save_base_dir": str(checkpoint_dir),
                            "save_dir": checkpoint_path.name,
                        }
                    }
                    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                        json.dump(override, handle)
                        override_path = Path(handle.name)

                    try:
                        val_metrics, test_metrics = show_entrypoint.print_search(
                            config_path=str(override_path),
                            default_config_path=str(suite["default_config"]),
                            args=_show_args(),
                        )
                    finally:
                        override_path.unlink(missing_ok=True)

                    self.assertNotEqual(
                        (val_metrics, test_metrics),
                        (0, 0),
                        msg=f"show.py could not load {checkpoint_path}",
                    )
                    self.assertTrue(val_metrics, msg=f"Missing validation metrics for {checkpoint_path}")

        if suites_run == 0:
            self.skipTest("No trained checkpoint suites were found on this machine.")


if __name__ == "__main__":
    unittest.main()
