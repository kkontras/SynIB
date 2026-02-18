import warnings
import os
from synib import mydatasets

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

from synib.utils.system.deterministic import deterministic
from synib.utils.system.misc import print_cuda_statistics
from synib.training.pipeline.helpers import (
    All_Evaluator,
    Loader,
    Monitor_n_Save,
    Trainer,
    Validator_Tester,
    pick_bias_infuser,
)

from synib.mydatasets import *

import os
import wandb
import torch.nn as nn
import logging
from accelerate import Accelerator, DistributedDataParallelKwargs

class Agent():
    @staticmethod
    def _resolve_dataloader_class(dataloader_name):
        dataloader_name = str(dataloader_name).strip()

        dataloader_cls = globals().get(dataloader_name, None)
        if dataloader_cls is None:
            dataloader_cls = getattr(mydatasets, dataloader_name, None)

        if dataloader_cls is None:
            available = sorted(
                name for name, value in vars(mydatasets).items() if isinstance(value, type)
            )
            raise KeyError(
                "Unknown dataloader class '{}'. Available dataloaders include: {}".format(
                    dataloader_name, ", ".join(available[:40])
                )
            )

        return dataloader_cls

    def __init__(self, config):
        self.config = config

        self.accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
            cpu=False,
        )
        deterministic(self.config.training_params.seed)

        if self.accelerator.is_main_process: print_cuda_statistics()

        dataloader = self._resolve_dataloader_class(self.config.dataset.dataloader_class)
        self.data_loader = dataloader(config=config)

        self.initialize_logs()
        self.get_loss()

        self.mem_loader = Loader(agent = self)
        self.monitor_n_saver = Monitor_n_Save(agent = self)
        self.trainer = Trainer(agent = self)
        self.validator_tester = Validator_Tester(agent = self)
        self.bias_infuser = pick_bias_infuser(agent = self)
        self.evaluators = All_Evaluator(self.config, dataloaders=self.data_loader)

        self.mem_loader.load_models_n_optimizer()
        self.mem_loader.get_scheduler()

        wandb.watch(self.model, log_freq=100)

    def initialize_logs(self):
        self.logger = logging.getLogger('Agent')
        self.logger.setLevel(logging.INFO)

        self.device = "cuda:{}".format(self.config.training_params.gpu_device[0])
        if self.accelerator.is_main_process: self.logger.info("Device: {}".format(self.device))

        self.steps_no_improve = 0
        if self.config.early_stopping.validate_every and self.config.early_stopping.end_of_epoch_check:
            max_steps = int(len(self.data_loader.train_loader) / self.config.early_stopping.validate_every) + 1

            if self.accelerator.is_main_process:
                self.logger.info("Total training batches: {}, validate every {} batches, steps per epoch: {}".format(
                    len(self.data_loader.train_loader), self.config.early_stopping.validate_every, max_steps))

        if "weights" in self.data_loader.train_loader.dataset.__dict__.keys():
            self.weights = self.data_loader.train_loader.dataset.weights
            self.config.model.args.class_weights = self.weights

        self.config.model.args.save_base_dir = self.config.model.save_base_dir

        if "weights" not in vars(self).keys(): self.weights = None

        best_logs_init = {"loss":{"total":100, "ce_loss_combined":100},
                               "acc":{"combined":0},
                               "pg_acc":{"combined":{"group_metrics":{"synergy":{"internal_acc":0}}}}}

        self.logs = {"current_epoch":0,"current_step":0,"steps_no_improve":0, "saved_step": 0, "train_logs":{},"val_logs":{},"test_logs":{},
                     "best_logs": {"best_vaccuracy":best_logs_init, "best_vloss":best_logs_init,  "best_vsyn_accuracy":best_logs_init},
                     "seed":self.config.training_params.seed, "weights": self.weights}
        if self.config.training_params.wandb_disable:
            self.wandb_run = wandb.init(reinit=True, project="synergy", config=self.config, mode = "disabled", name= self.config.model.save_dir.split("/")[-1][:-8])
        else:
            self.wandb_run = wandb.init(reinit=True, project="synergy", config=self.config, name= self.config.model.save_dir.split("/")[-1][:-8] )

    def get_loss(self):

        self.loss = nn.CrossEntropyLoss()

    def accelerate_components(self):
        self.model, self.optimizer, self.data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.data_loader, self.scheduler
        )

    def run(self):

        try:
            try:
                if not self.config.model.get("start_over", False):
                    self.mem_loader.load()
            except Exception as e:
                if "save_base_dir" in self.config.model:
                    file_name = os.path.join(self.config.model.save_base_dir, self.config.model.save_dir)
                else:
                    file_name = self.config.model.save_dir

                if os.path.exists(file_name):
                    self.logger.error("Error in loading the model: {}".format(e))

            self.accelerate_components()
            self.monitor_n_saver._early_stop_check_n_save(False)
            if self.evaluators.train_evaluator.get_early_stop(): return

            self.trainer.train_steps()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")
            return

    def finalize(self):

        self.logger.info("We are in the final state.")

        self.mem_loader.load_best_model()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.validator_tester.validate(test_set=False)
            best_val_metrics = self.evaluators.val_evaluator.evaluate()
            self.monitor_n_saver.print_valid_results(best_val_metrics, -1)

            if hasattr(self.data_loader, "test_loader"):
                self.validator_tester.validate(test_set=True)
                best_test_metrics = self.evaluators.test_evaluator.evaluate()
                self.monitor_n_saver.print_valid_results(best_test_metrics, -1, test=True)
                self.monitor_n_saver.save(model_save=False, verbose=True, post_test_results=best_test_metrics)
            else:
                self.monitor_n_saver.save(model_save=False, verbose=True, post_test_results=best_val_metrics)
