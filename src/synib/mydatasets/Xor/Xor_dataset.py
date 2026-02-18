from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import bernoulli
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class BinaryXORDataset(Dataset):
    """
    Generate n samples of data (v_a, v_b, ..., v_h) where
    v_a, v_b, ..., v_g ~ Bernoulli(0.5)
    v_h = v_a XOR v_b XOR v_c XOR ... XOR v_g
    """
    def __init__(self, config, mode, **kwargs):
        self.config = config
        self.n = self.config.dataset.n_points[mode]
        self.dim = self.config.dataset.dim
        self.p_hat = self.config.dataset.p_hat
        self.generate_data(self.n)

    def generate_data(self, n):
        # v_a = bernoulli.rvs(0.5, size=n)
        # v_b = bernoulli.rvs(0.5, size=n)
        # v_c = bernoulli.rvs(0.5, size=n)
        # v_d = bernoulli.rvs(0.5, size=n)
        # v_e = bernoulli.rvs(0.5, size=n)
        # v_f = bernoulli.rvs(0.5, size=n)
        # v_g = bernoulli.rvs(0.5, size=n)
        # # v_h = np.bitwise_xor.reduce([v_a, v_b, v_c, v_d, v_e, v_f, v_g])
        # v_h = np.bitwise_xor.reduce([v_a, v_b, v_c])
        # list_to_return = [torch.from_numpy(v).float().unsqueeze(1) for v in [v_a, v_b, v_c, v_d, v_e, v_f, v_g]]
        # list_to_return.append(torch.from_numpy(v_h).float())
        #
        # a_j, b_j ~ Bernoulli(0.5)
        self.a = torch.bernoulli(0.5 * torch.ones(n, self.dim))
        self.b = torch.bernoulli(0.5 * torch.ones(n, self.dim))
        self.i = torch.bernoulli(torch.full((n, 1), self.p_hat))
        xor_part = (self.a != self.b).float()
        self.c = xor_part * self.i + self.a * (1 - self.i)

        self.y = self.i.squeeze(-1).long()  # predict XOR vs copy
        # self.y = self.c


        # return list_to_return

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        # return {"data":{0:self.v_a[idx],
        #                 1:self.v_b[idx],
        #                 2:self.v_c[idx],
        #                 3:self.v_d[idx],
        #                 4:self.v_e[idx],
        #                 5:self.v_f[idx],
        #                 6:self.v_g[idx]}
        #     ,"label": self.v_h[idx], "idx": idx}
        #
        return {"data":{0:self.a[idx],
                        1:self.b[idx],
                        2:self.c[idx]}
            ,"label": self.y[idx], "idx": idx}


class NoisyBinaryXORDataset(Dataset):
    """
    Each sample: (a,b,c)
      with prob p_hat -> c = a XOR b (synergy case)
      else            -> c = a        (copy case)
    Label y = indicator of XOR vs copy
    Also adds a stable leak in a[:, -1] = y (train only)
    """
    def __init__(self, config,  mode="train"):
        super().__init__()

        self.config = config
        n_points = self.config.dataset.n_points[mode]
        dim = self.config.model.args.d_model
        p_hat = self.config.dataset.get("p_hat", 0.5)
        leak_prob = self.config.dataset.get("leak_prob", 1.0)
        self.n = n_points
        self.dim = dim
        self.p_hat = p_hat
        self.leak_prob = leak_prob
        self.mode = mode
        self._generate()

    def _generate(self):
        n, d = self.n, self.dim
        a = torch.bernoulli(0.5 * torch.ones(n, d))
        b = torch.bernoulli(0.5 * torch.ones(n, d))
        i = torch.bernoulli(torch.full((n, 1), self.p_hat))
        xor_part = (a != b).float()
        c = xor_part * i + a * (1 - i)
        y = i.squeeze(-1).long()

        # --- add stable label leak on training split ---
        if self.mode == "train" and self.leak_prob > 0:
            a[:, -1] = y.float()  # perfect shortcut
        else:
            a[:, -1] = torch.bernoulli(0.5 * torch.ones(n))  # noise

        self.a, self.b, self.c, self.y = a, b, c, y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "data": {0: self.a[idx], 1: self.b[idx], 2: self.c[idx]},
            "label": self.y[idx],
            "idx": idx,
        }

# class TriModalSynergyDataset(Dataset):
#     def __init__(self, n_samples=10000, latent_dim=8, mod_dim=16, noise=0.1, task="regression"):
#         """
#         A synthetic 3-modality dataset requiring cross-modal reasoning.
#         Returns dict: {
#             "data": {0: a_i, 1: b_i, 2: c_i},
#             "label": y_i,
#             "idx": i
#         }
#         """
#         super().__init__()
#         self.task = task
#
#         # Shared latent variable z
#         z = torch.randn(n_samples, latent_dim)
#
#         # Modality-specific projections
#         Wa, Wb, Wc = (
#             torch.randn(latent_dim, mod_dim),
#             torch.randn(latent_dim, mod_dim),
#             torch.randn(latent_dim, mod_dim),
#         )
#
#         a = torch.tanh(z @ Wa + noise * torch.randn(n_samples, mod_dim))
#         b = torch.tanh(z @ Wb + noise * torch.randn(n_samples, mod_dim))
#         c = torch.tanh(z @ Wc + noise * torch.randn(n_samples, mod_dim))
#
#         # Cross-modal interaction term — creates synergy
#         pairwise = a * b + b * c + c * a
#         Wy = torch.randn(mod_dim, 1)
#         y = (pairwise @ Wy).squeeze(-1)
#         y = torch.sigmoid(y + noise * torch.randn_like(y))
#
#         if task == "classification":
#             y = (y > 0.5).float()
#
#         self.a, self.b, self.c, self.y = a, b, c, y
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return {
#             "data": {
#                 0: self.a[idx],
#                 1: self.b[idx],
#                 2: self.c[idx],
#             },
#             "label": self.y[idx],
#             "idx": idx,
#         }


class TriModalSynergyDataset(Dataset):
    def __init__(self, config, mode, **kwargs):
        super().__init__()
        self.config = config
        self.mode = mode

        # Get parameters from config
        n_samples = self.config.dataset.n_points[mode]
        latent_dim = getattr(self.config.dataset, "latent_dim", 8)
        mod_dim = getattr(self.config.dataset, "mod_dim", 16)
        noise = getattr(self.config.dataset, "noise", 0.1)
        unimodal_weight = getattr(self.config.dataset, "unimodal_weight", 0.5)
        synergy_weight = getattr(self.config.dataset, "synergy_weight", 1.0)
        task = getattr(self.config.dataset, "task", "classification")

        z = torch.randn(n_samples, latent_dim)
        Wa, Wb, Wc = (
            torch.randn(latent_dim, mod_dim),
            torch.randn(latent_dim, mod_dim),
            torch.randn(latent_dim, mod_dim)
        )

        a = torch.tanh(z @ Wa + noise * torch.randn(n_samples, mod_dim))
        b = torch.tanh(z @ Wb + noise * torch.randn(n_samples, mod_dim))
        c = torch.tanh(z @ Wc + noise * torch.randn(n_samples, mod_dim))

        # Unimodal contributions (partial solvability)
        Ua = (a @ torch.randn(mod_dim, 1)).squeeze(-1)
        Ub = (b @ torch.randn(mod_dim, 1)).squeeze(-1)
        Uc = (c @ torch.randn(mod_dim, 1)).squeeze(-1)

        # Synergistic contributions (require cross-modal interactions)
        Uab = (a * b @ torch.randn(mod_dim, 1)).squeeze(-1)
        Ubc = (b * c @ torch.randn(mod_dim, 1)).squeeze(-1)
        Uca = (c * a @ torch.randn(mod_dim, 1)).squeeze(-1)

        y_cont = (
                unimodal_weight * (Ua + Ub + Uc)
                + synergy_weight * (Uab + Ubc + Uca)
                + noise * torch.randn(n_samples)
        )

        y = torch.sigmoid(y_cont)
        if task == "classification":
            y = (y > 0.5).float()

        self.a, self.b, self.c, self.y = a, b, c, y

    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        return {
            "data": {0: self.a[idx], 1: self.b[idx], 2: self.c[idx]},
            "label": self.y[idx],
            "idx": idx
        }


class BinaryXOR_Dataloader():

    def __init__(self, config):

        self.config = config

        dataset_train, dataset_val, dataset_test = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0))-1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))


        self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=num_cores,
                                                        shuffle=True,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        generator=g,
                                                        worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id))
        self.valid_loader = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_cores,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=num_cores,
                                                       pin_memory=self.config.training_params.pin_memory)

    def _get_datasets(self):


        # train_dataset = TriModalSynergyDataset(config=self.config, mode="train")
        # val_dataset = TriModalSynergyDataset(config=self.config, mode="val")
        # test_dataset = TriModalSynergyDataset(config=self.config, mode="test")

        train_dataset = NoisyBinaryXORDataset(config=self.config, mode="train")
        val_dataset = NoisyBinaryXORDataset(config=self.config, mode="val")
        test_dataset = NoisyBinaryXORDataset(config=self.config, mode="test")


        return train_dataset, val_dataset, test_dataset

