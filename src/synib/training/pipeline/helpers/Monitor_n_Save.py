import torch
from colorama import Fore
import wandb
import os
import os
import torch
from typing import Any, Dict, Optional
from colorama import Fore

VAL_WAYS = ["loss", "accuracy", "syn_accuracy"]

class Monitor_n_Save():

    def __init__(self, agent):
        self.agent = agent

    def save(
            self,
            verbose: bool = False,
            is_best_dict: Dict[str, bool] = None,
            model_save: bool = True,
            post_test_results: Optional[Dict[str, Any]] = None
    ) -> None:
        file_name = self._get_checkpoint_path()
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        if self.agent.logs["current_epoch"]<=1 and self.agent.config.model.get("start_over", False):
            existing_ckpt = None
        else:
            existing_ckpt = self._load_existing_checkpoint(file_name)

        checkpoint = self._build_base_checkpoint()

        self._add_model_state(
            checkpoint,
            existing_ckpt=existing_ckpt,
            model_save=model_save,
            is_best_dict=is_best_dict,
        )

        if post_test_results is not None:
            checkpoint["post_test_results"] = post_test_results
        if hasattr(self.agent.data_loader, "metrics"):
            checkpoint["metrics"] = self.agent.data_loader.metrics

        self._save_encoders(checkpoint)
        self._save_checkpoint(checkpoint, file_name, verbose)

    def _get_checkpoint_path(self) -> str:
        """Builds the model checkpoint path."""
        path = self.agent.config.model.save_dir

        if ("data_split" in self.agent.config.dataset and self.agent.config.dataset.data_split.get("split_method") == "patients_folds"):
            path = path.format(self.agent.config.dataset.data_split.fold)

        if hasattr(self.agent.config.model, "save_base_dir"):
            path = os.path.join(self.agent.config.model.save_base_dir, path)

        return path

    def _load_existing_checkpoint(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Loads existing checkpoint if it exists."""
        if os.path.exists(file_name):
            return torch.load(file_name, map_location="cpu", weights_only=False)
        return None

    def _build_base_checkpoint(self) -> Dict[str, Any]:
        """Creates the base checkpoint dict with optimizer, scheduler, logs, etc."""
        self.agent.logs["saved_step"] = self.agent.logs["current_step"]

        checkpoint = {
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "scheduler_state_dict": self.agent.scheduler.state_dict(),
            "logs": self.agent.logs,
            "configs": self.agent.config,
        }

        loader = getattr(self.agent.data_loader, "train_loader", None)
        if hasattr(loader, "generator") and loader.generator is not None:
            checkpoint["training_dataloader_generator_state"] = loader.generator.get_state()

        return checkpoint

    def _add_model_state(
            self,
            checkpoint: Dict[str, Any],
            existing_ckpt: Optional[Dict[str, Any]],
            model_save: bool,
            is_best_dict: Dict,
    ) -> None:
        """Adds model and best model states to the checkpoint."""
        unwrapped_model = self.agent.accelerator.unwrap_model(self.agent.model)

        list_to_check = ["best_model_state_dict"] + ["best_model_{}_state_dict".format(key) for key in VAL_WAYS]
        if model_save:
            checkpoint["model_state_dict"] = unwrapped_model.state_dict()
        else:
            list_to_check.append("model_state_dict")

        if existing_ckpt and not (self.agent.logs["current_epoch"]==1 and self.agent.config.model.get("start_over", False)):
            for key in list_to_check:
                if key in existing_ckpt:
                    checkpoint[key] = existing_ckpt[key]

        if is_best_dict is None:
            return

        for key, is_best in is_best_dict.items():
            if is_best:
                checkpoint["best_model_{}_state_dict".format(key)] = unwrapped_model.state_dict()
            else:
                if existing_ckpt and "best_model_{}_state_dict".format(key) in existing_ckpt:
                    checkpoint["best_model_{}_state_dict".format(key)] = existing_ckpt["best_model_{}_state_dict".format(key)]
                else:
                    checkpoint["best_model_{}_state_dict".format(key)] = unwrapped_model.state_dict()

    def _save_encoders(self, checkpoint: Dict[str, Any]) -> None:
        """Save encoders according to config."""
        config = self.agent.config
        encoders = getattr(config.model, "encoders", [])
        if not encoders: return

        fold = None
        if "data_split" in config.dataset and config.dataset.data_split.get("split_method") == "patients_folds":
            fold = config.dataset.data_split.fold

        for idx, enc_cfg in enumerate(encoders):
            if not enc_cfg.get("save_encoder", False): continue
            save_path_template = enc_cfg.get("pretrainedEncoder", {}).get("dir")
            if not save_path_template: continue

            base_dir = getattr(config.model, "save_base_dir", "")
            save_path = os.path.join(base_dir, save_path_template)
            if fold is not None: save_path = save_path.format(fold)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            encoder_attr = f"enc_{idx}"
            encoder = getattr(self.agent.model, encoder_attr, None)
            if encoder is None:
                self.agent.logger.warning(f"No encoder found for index {idx} ({encoder_attr}), skipping.")
                continue

            try:
                torch.save({"encoder_state_dict": encoder.state_dict()}, save_path)
                self.agent.logger.info(Fore.WHITE + f"Encoder {encoder_attr} saved successfully at: {save_path}"+ Fore.RESET)
            except Exception as e:
                self.agent.logger.error(f"Failed to save encoder {encoder_attr}: {e}")

    def _save_checkpoint(self, checkpoint: Dict[str, Any], file_name: str, verbose: bool) -> None:
        try:
            self.agent.accelerator.save(checkpoint, file_name)
            if verbose:
                self.agent.logger.info( Fore.WHITE + f"Model saved successfully at: {file_name}" + Fore.RESET)
        except Exception as e:
            raise RuntimeError(f"Error while saving model checkpoint: {e}") from e

    def monitoring(self):
        self.agent.accelerator.wait_for_everyone()
        if self.agent.accelerator.is_main_process:
            train_metrics = self.agent.evaluators.train_evaluator.evaluate()
            val_metrics = self.agent.evaluators.val_evaluator.evaluate(current_epoch=self.agent.logs["current_epoch"])

            self._find_learning_rate()

            self._update_train_val_logs(train_metrics = train_metrics, val_metrics = val_metrics)
            wandb.log({"train": train_metrics, "val": val_metrics}, step=self.agent.logs["current_step"]+1)

            is_best, is_best_dict = self.agent.evaluators.val_evaluator.is_best(metrics=val_metrics, best_logs=self.agent.logs["best_logs"])
            self.print_valid_results(val_metrics, self.agent.logs["current_step"])

            not_saved = True
            if is_best:

                self._update_best_logs(current_step = self.agent.logs["current_step"], val_metrics = val_metrics, is_best_dict=is_best_dict)

                wandb_out = {"best_val_{}".format(_key): val_metrics for _key in is_best_dict.keys() if is_best_dict[_key]}
                if self.agent.config.training_params.rec_test:
                    test_metrics = self._test_n_update()
                    wandb_out["test"] = test_metrics
                wandb.log(wandb_out, step=self.agent.logs["current_step"]+1)

                self.agent.logs["steps_no_improve"] = 0
                self.save(verbose = True, is_best_dict=is_best_dict)
                not_saved = False
            else:
                self.agent.logs["steps_no_improve"] += 1
                if self.agent.config.training_params.rec_test and self.agent.config.training_params.test_on_bottoms:
                    self._test_n_update()

            self._early_stop_check_n_save(not_saved)

    def _find_learning_rate(self):
        for param_group in self.agent.optimizer.param_groups:
            self.lr = param_group['lr']

    def _update_train_val_logs(self, train_metrics, val_metrics):

        train_metrics.update({  "validate_every": self.agent.config.early_stopping.validate_every,
                                "batch_size": self.agent.config.training_params.batch_size,
                                "learning_rate": self.agent.scheduler.lr_history[
                                                  max(self.agent.logs["current_step"] - self.agent.config.early_stopping.validate_every, 0):
                                                  self.agent.logs["current_step"]]})

        self.agent.logs["val_logs"][self.agent.logs["current_step"]] = val_metrics
        self.agent.logs["train_logs"][self.agent.logs["current_step"]] = train_metrics

    def _update_best_logs(self, current_step, val_metrics, is_best_dict):

        val_metrics.update({"step": current_step})
        for key in is_best_dict.keys():
            if is_best_dict[key]:
                self.agent.logs["best_logs"]["best_v{}".format(key)] = val_metrics

    def print_valid_results(self, val_metrics, current_step=None, test=False):

        if self.agent.config.training_params.verbose:
            if current_step is not None:
                step = int(current_step / self.agent.config.early_stopping.validate_every)
                message = Fore.WHITE + "Epoch {0:d} step {1:d} with ".format(self.agent.logs["current_epoch"], step)

            if not self.agent.config.training_params.tdqm_disable and not self.agent.trainer.end_of_epoch_check: print()

            if test:
                message = Fore.WHITE + "Test "
            else:
                message = Fore.WHITE + "Val "

            if current_step is not None:
                step = int(current_step / self.agent.config.early_stopping.validate_every)
                message += "Epoch {0:d} step {1:d} with ".format(self.agent.logs["current_epoch"], step)
            else:
                message += "Epoch {0:d} with ".format(self.agent.logs["current_epoch"])

            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i,v)
            if "acc" in val_metrics:
                for i, v in val_metrics["acc"].items():
                    if "best" not in i: message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i,v*100)
            if "synergy_gap_uni" in val_metrics:
                message += Fore.LIGHTBLUE_EX + "SyG_Uni: {:.2f} ".format(val_metrics["synergy_gap_uni"])
            if "synergy_gap_ens" in val_metrics:
                message += Fore.LIGHTBLUE_EX + "SyG_Ens: {:.2f} ".format(val_metrics["synergy_gap_ens"])
            if "f1" in val_metrics:
                for i, v in val_metrics["f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i,v*100)
            if "ceu" in val_metrics:
                for i, v in val_metrics["ceu"]["combined"].items(): message += Fore.LIGHTMAGENTA_EX + "CEU_{}: {:.4f} ".format(i,v)
            if "pg_acc" in val_metrics and "combined" in val_metrics["pg_acc"]:
                pg = val_metrics["pg_acc"]["combined"]
                m = pg.get('group_metrics', {})
                sl_dict = pg.get("synergy_per_label_acc", {})
                K = Fore.LIGHTWHITE_EX  # Main Keys (S, U1, U2, R)
                V = Fore.LIGHTYELLOW_EX  # Vivid Values (Accuracy)
                C = Fore.LIGHTRED_EX  # Contribution Values
                F1 = Fore.LIGHTBLUE_EX  # Synergy F1
                LK = Fore.LIGHTGREEN_EX  # Label Keys
                LV = Fore.LIGHTCYAN_EX  # Label Values
                N = Fore.RESET

                m = pg.get('group_metrics', {})
                sl_dict = pg.get("synergy_per_label_acc", {})
                syn_f1 = pg.get("synergy_f1", 0.0)

                s_str = f"{K}S:{V}{m['synergy']['internal_acc']:.1f}%{K}({C}{m['synergy']['contribution_to_total']:.1f}c{K}-{F1}F1:{syn_f1:.1f}%)"
                u1_str = f"{K}U1:{V}{m['cue_audio']['internal_acc']:.1f}%{K}({C}{m['cue_audio']['contribution_to_total']:.1f}c{K})"
                u2_str = f"{K}U2:{V}{m['cue_video']['internal_acc']:.1f}%{K}({C}{m['cue_video']['contribution_to_total']:.1f}c{K})"
                r_str = f"{K}R:{V}{m['coexistence']['internal_acc']:.1f}%{K}({C}{m['coexistence']['contribution_to_total']:.1f}c{K})"

                sl_items = [f"{LK}{k.split('_')[1]}:{LV}{v:.0f}%" for k, v in sl_dict.items() if v > 0]
                sl_str = f"{K}S-L: " + (" ".join(sl_items) if sl_items else "None")

                message += f"{s_str} {u1_str} {u2_str} {r_str} {sl_str}{N} "

            if self.agent.accelerator.is_main_process:
                self.agent.logger.info(message+ Fore.RESET)

    def _print_epoch_metrics(self):
        if self.agent.config.training_params.verbose:
            if self.agent.accelerator.is_main_process:
                self.agent.logger.info("Epoch {0:d}, N: {1:d}, lr: {2:.8f} Validation loss: {3:.6f}, accuracy: {4:.2f}% f1 :{5:.4f},  :{6:.4f}  Training loss: {7:.6f}, accuracy: {8:.2f}% f1 :{9:.4f}, k :{10:.4f},".format(
                    self.agent.logs["current_epoch"],
                    self.agent.logs["current_step"] * self.agent.config.training_params.batch_size * self.agent.config.dataset.seq_legth[0],
                    self.lr,
                    self.agent.logs["val_logs"][self.agent.logs["current_step"]]["loss"],
                    self.agent.logs["val_logs"][self.agent.logs["current_step"]]["acc"] * 100,
                    self.agent.logs["val_logs"][self.agent.logs["current_step"]]["f1"],
                    self.agent.logs["val_logs"][self.agent.logs["current_step"]]["k"],
                    self.agent.logs["train_logs"][self.agent.logs["current_step"]]["loss"],
                    self.agent.logs["train_logs"][self.agent.logs["current_step"]]["acc"] * 100,
                    self.agent.logs["train_logs"][self.agent.logs["current_step"]]["f1"],
                    self.agent.logs["train_logs"][self.agent.logs["current_step"]]["k"]))
    def _test_n_update(self):
        test_metrics = self.agent.evaluators.test_evaluator.evaluate()

        self.agent.logs["test_logs"][self.agent.logs["current_step"]] = test_metrics
        self.print_valid_results(test_metrics, self.agent.logs["current_step"], test=True)
        return test_metrics

    def print_test_results(self, val_metrics):

        if self.agent.config.training_params.verbose:
            message = Fore.WHITE + "Test "
            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items():
                    if "combined" in i:
                        message += Fore.RED + "{} : {:.6f} ".format(i, v)
            if "acc" in val_metrics:
                for i, v in val_metrics["acc"].items():
                    message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)

            if "acc_exzero" in val_metrics:
                for i, v in val_metrics[
                    "acc_exzero"].items(): message += Fore.LIGHTBLUE_EX + "Acc_ExZ_{}: {:.2f} ".format(i, v * 100)

            if "f1_perclass" in val_metrics:
                rounded_v = ["{:.1f}".format(v*100) for v in val_metrics["f1_perclass"]["combined"]]
                message += Fore.BLUE + "F1_perclass: {} ".format("-".join(rounded_v)) + Fore.RESET

            if self.agent.accelerator.is_main_process:
                self.agent.logger.info(message + Fore.RESET)

    def _early_stop_check_n_save(self, not_saved):

        training_cycle = (self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every)
        if not_saved and training_cycle % self.agent.config.early_stopping.save_every_valstep == 0:
            self.save()

        if training_cycle == self.agent.config.early_stopping.n_steps_stop_after:
            self.steps_at_earlystop_threshold = self.agent.logs["steps_no_improve"] # we dont need to initialize that since training_cycle > self.agent.config.n_steps_stop_after will not be true before ==

        if training_cycle > self.agent.config.early_stopping.n_steps_stop_after and self.agent.logs["steps_no_improve"] >= self.agent.config.early_stopping.n_steps_stop - 1:
            self.agent.evaluators.train_evaluator.set_early_stop()

