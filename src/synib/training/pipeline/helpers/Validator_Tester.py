import torch
from tqdm import tqdm
from collections import defaultdict
from synib.utils.data.flattendict import flatten_loss_dict
from synib.utils.data.to_device import to_device, to_float

class Validator_Tester():
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False

        task = self.agent.config.get("task", "classification")
        if task == "classification" or task == "bias_measure" or task == "generative":
            self.valtest_step_func = "valtest_one_step"
        elif task == "regression":
            self.valtest_step_func = "valtest_one_step_regression"
        elif task == "bias_measure":
            self.valtest_step_func = "valtest_one_step_biasmeasure"

        self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        self._get_loss_weights()

    def validate(self, test_set= False):

        self.agent.model.eval()

        this_evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        this_dataloader = self.agent.data_loader.test_loader if test_set else self.agent.data_loader.valid_loader
        this_evaluator.reset()
        with torch.no_grad():
            pbar = tqdm(enumerate(this_dataloader),
                        total=len(this_dataloader),
                        desc="Validation",
                        leave=False,
                        disable=True,
                        position=1)
            for batch_idx, served_dict in pbar:

                step_outcome = self.this_valtest_step_func(served_dict)

                all_outputs = self.agent.accelerator.gather(step_outcome)
                if all_outputs["process_flag"]:
                    this_evaluator.process(all_outputs)

                del step_outcome, served_dict

                mean_batch_loss, mean_batch_loss_message = this_evaluator.mean_batch_loss()

                pbar_message = "Validation batch {0:d}/{1:d} with {2:}".format(batch_idx,
                                                                             len(this_dataloader) - 1,
                                                                             mean_batch_loss_message)
                pbar.set_description(pbar_message)
                pbar.refresh()

    def valtest_one_step(self, served_dict):

            data = to_device(served_dict["data"], self.agent.accelerator.device)
            label = to_device(served_dict["label"], self.agent.accelerator.device)

            if len(label.shape) == 0: label = label.unsqueeze(dim=0)

            self.agent.model.eval()

            output = self.agent.model(data, label=label, current_epoch=self.agent.logs["current_epoch"])

            total_loss =  torch.zeros(1).squeeze().cuda()
            output_losses, ce_loss = {}, {}

            if self.agent.config.model.args.get("validation_loss_w", {"combined":1}):
                for k, v in output["preds"].items():
                    if k in self.agent.config.model.args.multi_loss.multi_supervised_w and self.agent.config.model.args.multi_loss.multi_supervised_w[k] != 0:
                        if len(label) > 1:  # TODO: Check if this one needs to be one or zero
                            if len(label.shape) > 1:
                                label = label.flatten()
                            ce_loss[k] = self.agent.loss(v, label)
                            total_loss += self.w_loss[k] * ce_loss[k]
                            ce_loss[k] = ce_loss[k].detach()
                            output_losses.update({"ce_loss_{}".format(k): ce_loss[k]})

            total_loss = total_loss.detach()
            if "losses" in output:
                output["losses"] = flatten_loss_dict(output["losses"])
                total_loss += sum(output["losses"].values())
                output_losses.update({i: output["losses"][i] for i in output["losses"]})
            output_losses.update({"total": total_loss})

            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach()

            process_flag = False
            if len(label) > 1:
                process_flag = True

            return {"loss": output_losses,
                    "pred" : output["preds"],
                    "features": output["features"],
                    "label": label,
                    "process_flag": process_flag}

    def valtest_one_step_regression(self, served_dict):

            data = {view: served_dict["data"][view].cuda() for view in
                                   served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor }
            data.update({view: data[view].float() for view in data if type(view) == int})

            label = served_dict["label"].cuda()

            output = self.agent.model(data)

            total_loss = torch.nn.L1Loss()(output["preds"]["combined"].flatten(), label)
            output_losses = {"l1_loss_combined": total_loss}

            total_loss = total_loss.detach()
            if "losses" in output:
                output["losses"] = flatten_loss_dict(output["losses"])
                total_loss += sum(output["losses"].values())
                output_losses.update({i: output["losses"][i] for i in output["losses"]})
            output_losses.update({"total": total_loss})
            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach()

            process_flag = False
            if len(label) > 1:
                process_flag = True

            return {"loss": output_losses,
                    "pred" : output["preds"],
                    "label": label,
                    "process_flag": process_flag}

    def _get_loss_weights(self):

        w_loss = defaultdict(int)
        w_loss["total"] = 1

        ws = self.agent.config.model.args.get("validation_loss_w", {"combined":1})
        for k, v in ws.items():
            w_loss[k] = v

        self.w_loss = w_loss

