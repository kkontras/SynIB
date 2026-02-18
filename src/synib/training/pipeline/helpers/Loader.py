import torch
import torch.nn as nn
from colorama import Fore
import torch.optim as optim
from synib.utils.schedulers.no_scheduler import No_Scheduler
from synib.utils.schedulers.warmup_scheduler import WarmupScheduler
import wandb
import torch._dynamo
import os
from synib.utils.optimization.normalized_adamw import NormalizedAdamW
from synib import models

import logging

TORCHDYNAMO_VERBOSE=1
TORCH_LOGS = "+dynamo"

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=False
try:
    torch._logging.set_logs(graph_breaks=True)
except Exception:
    pass

logger = logging.getLogger('torch._dynamo.symbolic_convert:')
logger.setLevel(logging.WARNING)

logger = logging.getLogger('torch._dynamo.output_graph:')
logger.setLevel(logging.WARNING)

class Loader():

    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def _resolve_model_class(model_name):
        model_name = str(model_name).strip()
        model_cls = getattr(models, model_name, None)
        if model_cls is None:
            available = sorted(
                name for name, value in vars(models).items() if isinstance(value, type)
            )
            raise KeyError(
                "Unknown model class '{}'. Available model classes include: {}".format(
                    model_name, ", ".join(available[:60])
                )
            )
        return model_cls

    def _resolve_checkpoint_path(self, file_path):
        if not file_path:
            return file_path
        if os.path.isabs(file_path):
            return file_path
        if "save_base_dir" in self.agent.config.model:
            return os.path.join(self.agent.config.model.save_base_dir, file_path)
        return file_path

    @staticmethod
    def _safe_torch_load(file_path, map_location=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(file_path))
        return torch.load(file_path, map_location=map_location, weights_only=False)

    def load_pretrained_models(self):
        if "pretrained_model" in self.agent.config.model:
            if self.agent.config.model.pretrained_model["use"] and not self.agent.config.model.load_ongoing:
                if self.agent.accelerator.is_main_process:
                    self.agent.logger.info("Loading pretrained model from file {}".format(self.agent.config.model.pretrained_model["dir"]))
                checkpoint_path = self._resolve_checkpoint_path(self.agent.config.model.pretrained_model["dir"])
                checkpoint = self._safe_torch_load(checkpoint_path, map_location="cpu")
                self.agent.model.load_state_dict(checkpoint["model_state_dict"])

    def _freeze_encoders(self, config_model, model):
        for enc in range(len(config_model.get("encoders", []))):
            enc_args = config_model.encoders[enc].get("args",{})
            if enc_args.get("freeze_encoder", False):
                if hasattr(model, "enc_{}".format(enc)):
                    self.agent.logger.info("Freezing encoder enc_{}".format(enc))
                    for p in getattr(model, "enc_{}".format(enc)).parameters():
                        p.requires_grad = False
                    for m in getattr(model, "enc_{}".format(enc)).modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.track_running_stats = False

            if "encoders" in config_model.encoders[enc]:
                for enc_i in range(len(config_model.encoders)):
                    self._freeze_encoders(config_model = config_model.encoders[enc_i], model = getattr(model, "enc_{}".format(enc_i)))

    def keep_only_trainable_params(self, list_of_params):
        params = []
        for p in list_of_params:
            if isinstance(p, list):
                for p_i in p:
                    if p_i.requires_grad:
                        params.append(p_i)
                continue
            else:
                if p.requires_grad:
                    params.append(p)
        return params

    def load_models_n_optimizer(self):

        enc = self.load_encoder(enc_args=self.agent.config.model.get("encoders", []))
        model_class = self._resolve_model_class(self.agent.config.model.model_class)

        if "save_base_dir" in self.agent.config.model and "swin_backbone" in self.agent.config.model.args:
            self.agent.config.model.args.swin_backbone = os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.args.swin_backbone)

        if "save_base_dir" in self.agent.config.model and "pretraining_paths" in self.agent.config.model.args:
            self.agent.config.model.args.pretraining_paths = {i: os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.args.pretraining_paths[i]) for i in self.agent.config.model.args.pretraining_paths}

        self.agent.model = model_class(encs = enc, args = self.agent.config.model.args)
        self._freeze_encoders(config_model=self.agent.config.model, model=self.agent.model)
        self.agent.model.cuda()

        self._my_numel(self.agent.model, verbose=True)

        if self.agent.config.optimizer.type == "Adam":
            ind_opt = self.agent.config.optimizer.get("indepentent_params", False)
            all_param_names = [name for name, _ in self.agent.model.named_parameters()]
            if ind_opt is not False:
                list_of_params = []
                for i, key in enumerate(ind_opt):
                    if hasattr(self.agent.model, key):
                        this_param_names = ["{}.{}".format(key,name) for name, _ in getattr(self.agent.model, key).named_parameters()]
                        all_param_names = [name for name in all_param_names if name not in this_param_names]
                        list_of_params.append({'params':getattr(self.agent.model, key).parameters(), "lr": ind_opt[key]["learning_rate"], "weight_decay": ind_opt[key]["weight_decay"]})
                list_of_params.append({'params': [p for n_i in all_param_names for name, p in self.agent.model.named_parameters() if name == n_i], "lr": self.agent.config.optimizer.learning_rate, "weight_decay": self.agent.config.optimizer.weight_decay})
                list_of_params = self.keep_only_trainable_params(list_of_params)
                self.agent.optimizer = optim.Adam(list_of_params,
                                                  lr=self.agent.config.optimizer.learning_rate,
                                                  betas=(self.agent.config.optimizer.beta1, self.agent.config.optimizer.beta2),
                                                  eps=1e-07,
                                                  weight_decay=self.agent.config.optimizer.weight_decay)
            else:
                list_of_params = self.keep_only_trainable_params(self.agent.model.parameters())
                self.agent.optimizer = optim.Adam(list_of_params,
                                                  lr=self.agent.config.optimizer.learning_rate,
                                                  betas=(self.agent.config.optimizer.beta1, self.agent.config.optimizer.beta2),
                                                  eps=1e-07,
                                                  weight_decay=self.agent.config.optimizer.weight_decay)
        elif self.agent.config.optimizer.type == "SGD":
            self.agent.optimizer = optim.SGD(self.agent.model.parameters(),
                                    lr=self.agent.config.optimizer.learning_rate,
                                    weight_decay=self.agent.config.optimizer.weight_decay,
                                    momentum=self.agent.config.optimizer.momentum)
        elif self.agent.config.optimizer.type == "Adadelta":
            self.agent.optimizer = optim.Adadelta(self.agent.model.parameters(),
                                            lr=self.agent.config.optimizer.learning_rate,
                                            rho=0.9,
                                            eps=1e-06,
                                            weight_decay=self.agent.config.optimizer.weight_decay)
        elif self.agent.config.optimizer.type == "Adaw":
            ind_opt = self.agent.config.optimizer.get("indepentent_params", False)
            if ind_opt is not False:
                list_of_params = []
                for i, key in enumerate(ind_opt):
                    name = "mod{}_{}_model".format(i, key)
                    if hasattr(self.agent.model, name):
                        list_of_params.append({'params':getattr(self.agent.model, name).parameters(), "lr": ind_opt[key]["learning_rate"], "weight_decay": ind_opt[key]["weight_decay"]})
                if hasattr(self.agent.model, "classifier"):
                    list_of_params.append({'params':self.agent.model.classifier.parameters(), "lr": self.agent.config.optimizer.learning_rate, "weight_decay": self.agent.config.optimizer.weight_decay})
                self.agent.optimizer = optim.AdamW(list_of_params,
                                            lr=self.agent.config.optimizer.learning_rate,
                                            weight_decay=self.agent.config.optimizer.weight_decay)
            else:
                self.agent.optimizer = optim.AdamW(self.agent.model.parameters(),
                                            lr=self.agent.config.optimizer.learning_rate,
                                            weight_decay=self.agent.config.optimizer.weight_decay)
        elif self.agent.config.optimizer.type == "NormalizedAdamW":
            self.agent.optimizer = NormalizedAdamW(self.agent.model.parameters(),
                                        lr=self.agent.config.optimizer.learning_rate,
                                        weight_decay=self.agent.config.optimizer.weight_decay)
        self.load_pretrained_models()

    def load_best_model(self):

        file_name = self.agent.config.model.save_dir
        if "data_split" in self.agent.config.dataset and self.agent.config.dataset.data_split.get("split_method",
                                                                                                  False) == "patients_folds":
            file_name = file_name.format(self.agent.config.dataset.data_split.fold)

        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, file_name)

        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
            val_with = self.agent.config.early_stopping.get("validate_with", "accuracy")
            if "best_model_state_dict" in prev_checkpoint:

                self.agent.model.load_state_dict(prev_checkpoint["best_model_state_dict"])
                logging.info("Loaded best model from {}".format(file_name))
            elif "best_model_{}_state_dict".format(val_with) in prev_checkpoint:
                logging.info("Loaded best model from {}".format("best_model_{}_state_dict".format(val_with)))
                self.agent.model.load_state_dict(prev_checkpoint["best_model_{}_state_dict".format(val_with)])
        else:
            logging.info("No file found in {}".format(file_name))

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose = True):

        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params =  sum(p.numel() for p in unique)
        if verbose and self.agent.accelerator.is_main_process:
            self.agent.logger.info("Total number of trainable parameters are: {}".format(model_total_params))

        return model_total_params

    def get_scheduler(self):
        if self.agent.config.scheduler.type == "cyclic":
            after_scheduler = optim.lr_scheduler.CyclicLR(self.agent.optimizer, base_lr=self.agent.config.optimizer.learning_rate, max_lr=self.agent.config.scheduler.max_lr, cycle_momentum=False)

            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)

        elif self.agent.config.scheduler.type == "cosanneal":

            after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.agent.optimizer, T_0=4, T_mult=2)
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)

        elif self.agent.config.scheduler.type == "reducerlonplatau":

            after_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.agent.optimizer,
                                                                   mode='min',
                                                                   factor=self.agent.config.scheduler.factor,
                                                                   patience=self.agent.config.scheduler.patience,
                                                                   verbose=True
                                                                   )
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)
        elif self.agent.config.scheduler.type == "stepLR":
            after_scheduler = optim.lr_scheduler.StepLR(optimizer=self.agent.optimizer, step_size=self.agent.config.scheduler.lr_decay_step, gamma=self.agent.config.scheduler.lr_decay_ratio)
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)
        elif self.agent.config.scheduler.type == "MultistepLR":
            after_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.agent.optimizer, milestones=self.agent.config.scheduler.milestones, gamma=self.agent.config.scheduler.lr_decay_ratio)
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)

        else:
            self.agent.scheduler = No_Scheduler(base_lr=self.agent.config.optimizer.learning_rate)

    def load(self):

        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.save_dir)
        else:
            file_name = self.agent.config.model.save_dir

        if not os.path.exists(file_name):
            return False

        checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)

        if self.agent.accelerator.is_main_process:
            self.agent.logger.info("Loading checkpoint: {}".format(file_name))

        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if "training_dataloder_generator_state" in checkpoint:
            self.agent.data_loader.train_loader.generator.set_state(checkpoint["training_dataloder_generator_state"])
        if "logs" in checkpoint:
            self.agent.logs = checkpoint["logs"]

        if hasattr(checkpoint, "metrics"):
            self.agent.data_loader.load_metrics_ongoing(checkpoint["metrics"])

        if hasattr(self.agent.logs, "weights"):
            self.agent.data_loader.weights = self.agent.logs["weights"]
            self.agent.weights = self.agent.data_loader.weights
            self.agent.logger.info("Loaded loss weights are:", self.agent.weights)

        for step in self.agent.logs["train_logs"]:
            wandb.log({"train": self.agent.logs["train_logs"][step], "val":  self.agent.logs["val_logs"][step]}, step=step)
            for i, lr in enumerate(self.agent.logs["train_logs"][step]["learning_rate"]):
                wandb.log({"lr": lr, "val":  self.agent.logs["val_logs"][step]}, step=i+ step - self.agent.config.early_stopping.validate_every)

        self.agent.loss = nn.CrossEntropyLoss()

        val_with = self.agent.config.early_stopping.validate_with
        this_best_log = self.agent.logs["best_logs"]["best_v{}".format(val_with)]

        message = ""
        if "step" in this_best_log:
            message += Fore.WHITE + "The best in step: {} so far \n".format(
                int(this_best_log["step"] / self.agent.config.early_stopping.validate_every))
            if "loss" in this_best_log:
                for i, v in this_best_log["loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i,v)
            if "acc" in this_best_log:
                for i, v in this_best_log["acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
            if "f1" in this_best_log:
                for i, v in this_best_log["f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
            if "k" in this_best_log:
                for i, v in this_best_log["k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)

        if self.agent.accelerator.is_main_process:
            self.agent.logger.info("Checkpoint loaded successfully")
            print(self.agent.logs["best_logs"])
            self.agent.logger.info(message+Fore.RESET)

    def load_encoder(self, enc_args):
        encs = []
        for num_enc in range(len(enc_args)):
            enc_class = self._resolve_model_class(enc_args[num_enc]["model"])
            args = enc_args[num_enc]["args"]

            if "encoders" in enc_args[num_enc]:
                enc_enc = self.load_encoder(enc_args[num_enc]["encoders"])
                enc = enc_class(encs = enc_enc, args = args)
            else:
                enc = enc_class(args = args, encs=[])

            pretrained_encoder_args =  enc_args[num_enc].get("pretrainedEncoder", {"use":False})
            if pretrained_encoder_args["use"]:

                file_path = self._resolve_checkpoint_path(pretrained_encoder_args.get("dir",""))
                val_with = self.agent.config.early_stopping.get("validate_with", "accuracy")
                checkpoint = self._safe_torch_load(file_path, map_location="cpu")
                if "encoder_state_dict" in checkpoint:
                    missing_keys, unexpected_keys =  enc.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
                    if missing_keys or unexpected_keys:
                        logging.warn(f"Missing keys in state_dict: {missing_keys}")
                        logging.warn(f"Unexpected keys in state_dict: {unexpected_keys}")
                    else:
                        logging.info("Loading enc encoder state dict from {}".format(file_path))

                elif "best_model_state_dict" in checkpoint:
                    logging.info("Loading enc best model state dict from {}".format(file_path))
                    missing_keys, unexpected_keys =  enc.load_state_dict(checkpoint["best_model_state_dict"], strict=False)
                    if missing_keys or unexpected_keys:
                        logging.warn(f"Missing keys in state_dict: {missing_keys}")
                        logging.warn(f"Unexpected keys in state_dict: {unexpected_keys}")

                elif "best_model_{}_state_dict".format(val_with) in checkpoint:
                    logging.info("Loading enc best model state dict from {}".format(file_path))
                    missing_keys, unexpected_keys = enc.load_state_dict(checkpoint["best_model_{}_state_dict".format(val_with)])
                    if missing_keys or unexpected_keys:
                        logging.warn(f"Missing keys in state_dict: {missing_keys}")
                        logging.warn(f"Unexpected keys in state_dict: {unexpected_keys}")

            encs.append(enc)
        return encs
