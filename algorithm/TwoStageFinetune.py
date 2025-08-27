import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adam, lr_scheduler
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import albumentations as A
from typing import Dict, List, Tuple
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, dataset
from FeatureUnlearn import ImageBias

from utlis import *

import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
setup_seed(42)

class ParameterPerturber:
    def __init__(
            self,
            model,
            opt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device

        self.weight_lower_bound = 1

    @staticmethod
    def get_layer_num(layer_name: str) -> int:
        # get the whole number of network layers
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    @staticmethod
    def zerolike_params_dict(model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter values
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params

        Set all parameters to 0 and make it a dict
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    @staticmethod
    def randomlike_params_dict(model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter values
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params

        Set all parameters to 0 and make it a dict
        """
        return dict(
            [
                (k, torch.randn_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    @staticmethod
    def oneslike_params_dict(model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter values
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params

        Set all parameters to 0 and make it a dict
        """
        return dict(
            [
                (k, torch.ones_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    @staticmethod
    def calculate_class_weight(dataset: dataset, n_class, pos_weight):
        subset_idxs = [[] for _ in range(n_class)]
        for idx, (x, y, z) in enumerate(dataset):
            subset_idxs[y.int()].append(idx)
        class_counts = [len(subset_idxs[idx]) for idx in range(n_class)]
        if pos_weight:
            weights = torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float32)
            # weights = torch.broadcast_to(weights, args.batch_size)
        else:
            sum_counts = sum(class_counts)
            class_freq = []
            for i in class_counts:
                class_freq.append(i / sum_counts * 100)
            weights = torch.tensor(class_freq, dtype=torch.float32)

            weights = weights / weights.sum()
            weights = 1.0 / weights
            weights = weights / weights.sum()
            weights = weights.to(device)

        return weights

    @staticmethod
    def get_block_importance(importance):
        layer_name = []
        imp_block_tensor = []
        layer_count = 0
        block_count = 0
        block_layers = resnet_block_definition()
        block_importance = {}
        for imp_n, imp in importance.items():
            layer_name.append(imp_n)
            layer_count += 1
            imp_block_tensor.append(imp.view(-1))
            if layer_count == block_layers[block_count]:
                layer_count = 0
                block_count += 1
                block_importance[tuple(layer_name)] = torch.mean(torch.concat(imp_block_tensor))
                imp_block_tensor = []
                layer_name = []
        return block_importance

    def calc_importance(self, dataloader: DataLoader, metric='bias', mask_type='fisher') -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        importances = self.zerolike_params_dict(self.model)
        for data in dataloader:
            inputs, labels, attrs = data
            inputs, labels, attrs = inputs.to(self.device), labels.to(self.device), attrs.to(self.device)
            self.opt.zero_grad()

            X = inputs
            y = labels.to(torch.float)
            p = attrs
            # criterion = nn.BCEWithLogitsLoss()

            out = self.model(X).squeeze(1)
            count_pos = torch.sum(y) * 1.0 + 1e-10
            count_neg = torch.sum(1. - y) * 1.0
            beta = count_neg / count_pos
            beta_back = count_pos / (count_pos + count_neg)
            bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
            if metric == 'bias':
                out = torch.sigmoid(out)
                # loss = compute_empirical_bias(out, y, p, args.bias)
                loss = diff_bias_loss(out, y, p, args.bias)
            elif metric == 'loss_bias':
                # loss = args.beta * beta_back * bce1(out, y) + compute_empirical_bias(out, y, p,
                #                                                                      args.bias)
                loss = args.beta * beta_back * bce1(out, y) + compute_empirical_bias(torch.sigmoid(out), y, p, args.bias)
            else:
                loss = beta_back * bce1(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    if mask_type == 'fisher':
                        imp.data += p.grad.data.clone().pow(2)
                    else:
                        imp.data += p.grad.data.clone()

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def generate_mask(
            self,
            para_importance: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """
        with torch.no_grad():
            mask = self.zerolike_params_dict(self.model)
            for (imp_n, imp_p) in para_importance.items():
                mask[imp_n] = torch.tanh(imp_p)
        return mask

    def cal_imp_diff(self,
                     group1_importance: Dict[str, torch.Tensor],
                     group2_importance: Dict[str, torch.Tensor]
                     ):
        with torch.no_grad():
            importance = self.zerolike_params_dict(self.model)
            for (imp_n1, imp1), (imp_n2, imp2) in zip(
                    group1_importance.items(),
                    group2_importance.items(),
            ):
                if (torch.max(imp1) - torch.min(imp1)) == 0:
                    imp1 = imp1
                else:
                    imp1 = self.minmaxnorm(imp1)
                if (torch.max(imp2) - torch.min(imp2)) == 0:
                    imp2 = imp2
                else:
                    imp2 = self.minmaxnorm(imp2)
                importance[imp_n1] = imp1 / (imp2 + 1e-10)
        return importance

    @staticmethod
    def calculate_threshold(importance: Dict[str, torch.Tensor], rate=None) -> float:
        imp_tensor = []
        with torch.no_grad():
            for imp_n, imp in importance.items():
                # if 'conv' in imp_n:
                # if ('downsample' not in imp_n) & ('head' not in imp_n):
                if 'head' not in imp_n:
                    if args.mask_scale == 'weight':
                        imp_tensor.append(imp.view(-1))
                    elif args.mask_scale == 'layer':
                        imp_tensor.append(torch.mean(imp.view(-1)).unsqueeze(-1))
                    else:
                        imp_tensor.append(imp.unsqueeze(-1))

            if rate is None:
                threshold = torch.mean(torch.cat(imp_tensor))
            else:
                threshold = np.percentile(torch.cat(imp_tensor).cpu().numpy(), rate)

        return threshold

    @staticmethod
    def calculate_minmax(importance: Dict[str, torch.Tensor], component=None):
        imp_tensor = []
        with torch.no_grad():
            for imp_n, imp in importance.items():
                if component is not None:
                    if component in imp_n:
                # if 'conv' in imp_n:
                #     if ('layer2' in imp_n) | ('layer3' in imp_n):
                        imp_tensor.append(imp.view(-1))
                else:
                    imp_tensor.append(imp.view(-1))
            total_imp = torch.cat(imp_tensor)
            min_value = torch.min(total_imp)
            max_value = torch.max(total_imp)
            mean_value = torch.mean(total_imp)
            std_value = torch.std(total_imp)
        return min_value, max_value, mean_value, std_value

    @staticmethod
    def calculate_sum(importance: Dict[str, torch.Tensor], rate=None):
        imp_tensor = []
        with torch.no_grad():
            for imp_n, imp in importance.items():
                if 'head' not in imp_n:
                    imp_tensor.append(imp.view(-1))
            total_imp = torch.cat(imp_tensor)
            sum_value = torch.sum(total_imp)
        return sum_value

    @classmethod
    def standardize(cls, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        x = x.view(-1)

        ret = (x - x.mean()) / x.std()

        return ret.view(*sh)

    @classmethod
    def minmaxnorm(cls, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        x = x.view(-1)

        ret = (x - x.min()) / (x.max() - x.min() + 1e-10)

        return ret.view(*sh)

    @classmethod
    def percentile_normalize(cls, vec, low=0.0, high=100.0, eps=1e-10):
        """
        Robust percentile-based normalisation.
        - `low`, `high` are the percentiles that define the stretch
          (defaults → full range 0–100).
        - Values below the `low`-th percentile map to 0, above the
          `high`-th map to 1, everything else is linearly scaled.
        """
        if vec.numel() <= 1:  # nothing to scale
            return vec

        p_low = torch.quantile(vec, low / 100.0)
        p_high = torch.quantile(vec, high / 100.0)

        denom = (p_high - p_low).clamp(min=eps)  # avoid divide-by-0
        scaled = (vec - p_low) / denom
        return scaled.clamp_(0.0, 1.0)


class Debiasing:
    def __init__(
            self,
            model,
            dataset,
            pos_weight=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.weight_lower_bound = 1
        self.pos_weight = pos_weight
        if args.task == 'skin':
            self.optimizer = SGD(model.parameters(), lr=args.lr_forget, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = Adam(self.model.net.parameters(), lr=args.lr_forget)
        self.pdr = ParameterPerturber(self.model, self.optimizer, self.device)

    def finetune(self, with_l1=False, mask=None, loss_metric='loss', dataloader=None):
        epochs = args.n_epochs
        total_step = int(len(dataloader) * epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(len(dataloader) * epochs / 3), gamma=0.1)
        # total_step = epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_step)
        self.model.train()
        for ep in range(epochs):
            print('Epochs {} start:--------------------------'.format(ep))
            mean_loss = 0
            for samples in dataloader:
                inputs, targets, attribute = samples
                inputs, targets, attribute = inputs.to(self.device), targets.to(self.device), attribute.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                # BCE weighted loss
                count_pos = torch.sum(targets) * 1.0 + 1e-10
                count_neg = torch.sum(1. - targets) * 1.0
                beta = count_neg / count_pos
                beta_back = count_pos / (count_pos + count_neg)
                bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
                bceloss = beta_back * bce1(outputs, targets)
                biasloss = diff_bias_loss(torch.sigmoid(outputs), targets, attribute, args.bias)
                if loss_metric == 'bias':
                    outputs = torch.sigmoid(outputs)
                    loss = diff_bias_loss(outputs, targets, attribute, args.bias)
                elif loss_metric == 'loss_bias':
                    lamda_reg = 1
                    loss = args.beta * bceloss + (1 - args.beta) * lamda_reg * biasloss
                elif loss_metric == 'loss':
                    loss = bceloss
                else:
                    loss = bce1(outputs, targets)
                mean_loss += loss
                loss.backward()
                if with_l1:
                    loss += args.alpha * l1_regularization(self.model)
                if mask:
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n]
                self.optimizer.step()
                # scheduler.step()
            mean_loss = mean_loss / len(dataloader)
            print('Loss is {}-------------'.format(mean_loss))

        net = self.model
        return net

    def bias_tuning(self, with_l1=False, mask=None, dataloader=None):
        finetune_epochs = args.unlearn_epochs
        total_steps = finetune_epochs * len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.model.train()
        for ep in range(finetune_epochs):
            mean_loss = 0
            for samples in dataloader:
                inputs, targets, attribute = samples
                inputs, targets, attribute = inputs.to(self.device), targets.to(self.device), attribute.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                count_pos = torch.sum(targets) * 1.0 + 1e-10
                count_neg = torch.sum(1. - targets) * 1.0
                beta = count_neg / count_pos
                beta_back = count_pos / (count_pos + count_neg)
                bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
                bceloss = beta_back * bce1(outputs, targets)
                biasloss = diff_bias_loss(torch.sigmoid(outputs), targets, attribute, args.bias)
                lamda_reg = 1
                loss = args.beta * bceloss + (1 - args.beta) * lamda_reg * biasloss
                loss.backward()
                mean_loss += loss

                if with_l1:
                    loss += args.alpha * l1_regularization(self.model)
                if mask:
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n]
                self.optimizer.step()
                # scheduler.step()
            mean_loss = mean_loss / len(dataloader)
            print('Ep {}: Loss is {}-------------'.format(ep, mean_loss))

        net = self.model
        return net

    def mask_select(self, dataloader, metric, mask_type=None):
        if mask_type == 'fisher':
            bias_importance = self.pdr.calc_importance(dataloader, metric=metric, mask_type=mask_type)
        elif mask_type == 'grad':
            bias_importance = self.pdr.calc_importance(dataloader, metric=metric, mask_type=mask_type)
        elif mask_type == 'group':
            csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
            group0_csv = csv[csv[args.attr] == 0].reset_index(drop=True)
            group1_csv = csv[csv[args.attr] == 1].reset_index(drop=True)
            group0_dataset = get_dataset(group0_csv, args.attr, transform=None, mode='test')
            group1_dataset = get_dataset(group1_csv, args.attr, transform=None, mode='test')
            group0_dataloader = load_dataset(group0_dataset, shuffle=True)
            group1_dataloader = load_dataset(group1_dataset, shuffle=True)
            group0_imp = self.pdr.calc_importance(dataloader=group0_dataloader, metric=metric, mask_type='fisher')
            group1_imp = self.pdr.calc_importance(dataloader=group1_dataloader, metric=metric, mask_type='fisher')
            bias_importance = self.pdr.cal_imp_diff(group0_imp, group1_imp)

        elif mask_type == 'random':
            bias_importance = self.pdr.randomlike_params_dict(self.model)
        else:
            bias_importance = self.pdr.oneslike_params_dict(self.model)

        return bias_importance

    def fisher_mask_debiasing(self, loss_metric=None, mask_type=None, state_dict=None, subset=0):
        dataloader = load_dataset(self.dataset, shuffle=True)
        if args.task == 'skin':
            self.optimizer = SGD(self.model.head.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=args.lr_base)
        net = self.finetune(with_l1=False, mask=None, loss_metric=loss_metric, dataloader=dataloader)
        return net

    def impair_repair_debiasing(self, loss_metric=None, mask_type=None, subset=0):
        # dataloader = load_dataset(self.dataset, shuffle=True)
        csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
        self.model.eval()
        ## AUC gap calculation for multiple attributes
        if args.num_attr == 'multi':
            dataloader = load_dataset(self.dataset, shuffle=False)
            max_group, min_group, max_auc, min_auc, auc_gap = auc_gap_calc(dataloader=dataloader, output_csv=csv, net=self.model, attr=args.attr)
            csv = csv[(csv[args.attr] == max_group) | (csv[args.attr] == min_group)]
            csv = csv.reset_index(drop=True)
            csv[args.attr] = csv[args.attr].apply(lambda x: 1 if x == min_group else 0)

        ft_dataset = get_dataset(csv, args.attr, transform=None, mode='test')
        dataloader = load_dataset(ft_dataset, batch_size=args.batch_size, shuffle=True)

        bias_importance = self.mask_select(dataloader, metric='bias', mask_type=mask_type)
        pred_importance = self.mask_select(dataloader, metric='loss', mask_type=mask_type)
        bias_importance = self.pdr.cal_imp_diff(bias_importance, pred_importance)
        # bias_importance = self.mask_select(dataloader, metric='loss', mask_type='group')
        bias_mask = self.pdr.generate_mask(bias_importance)
        # mask_min, mask_max, mask_mean, mask_std = self.pdr.calculate_minmax(bias_mask, 'head')
        # print(f'The mean value of the mask {mask_mean}')

        # Repair
        # csv = target_dataset_balance(csv, 'target')
        csv = dataset_balance(csv, args.attr, 'target', 1)
        dataset = get_dataset(csv, args.attr, transform=None, mode='train')
        dataloader = load_dataset(dataset, batch_size=args.batch_size, shuffle=True)
        for (net_n, net_p) in self.model.named_parameters():
            if 'head' in net_n:
                net_p.requires_grad = False
            else:
                net_p.requires_grad = True
        if args.task == 'skin':
            self.optimizer = SGD(self.model.net.parameters(), lr=args.lr_forget, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = Adam(self.model.net.parameters(), lr=args.lr_forget)
        net = self.bias_tuning(with_l1=False, mask=bias_mask, dataloader=dataloader)
        # net = self.model
        net.eval()
        with torch.no_grad():
            for (net_n, net_p), (imp_n, imp_p) in zip(
                    net.named_parameters(),
                    bias_mask.items(),
            ):
                if 'head' in net_n:
                    thresh = np.mean(imp_p.view(-1).cpu().numpy())
                    locations = torch.where(imp_p > thresh)
                    net_p[locations] = 0

                    net_p.requires_grad = True
                    # print((net_p==0).sum().item()/net_p.numel())

                else:
                    net_p.requires_grad = False

        args.beta = 1 - args.beta
        if args.task == 'skin':
            self.optimizer = SGD(self.model.head.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = Adam(self.model.head.parameters(), lr=args.lr_base)

        net = self.finetune(with_l1=False, mask=None, loss_metric=loss_metric, dataloader=dataloader)

        return net


def main():
    setup_seed(42)
    checkpoint = torch.load(args.model_dir, map_location=device)
    state_dict = checkpoint['state_dict']
    csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)

    dataset = get_dataset(csv, args.attr, transform=None, mode='test')
    net = load_model(state_dict)
    debiasing_module = Debiasing(net, dataset, device=device)
    net = debiasing_module.impair_repair_debiasing(loss_metric='loss_bias', mask_type='fisher', subset=0)

    if args.task == 'skin':
        if args.attr == 'skin_attribute':
            test_csv_dir = './data/skin/csv/fitzpatrick17k.csv'
            test_csv = pd.read_csv(test_csv_dir, low_memory=False).reset_index(drop=True)
            test_dataset = get_dataset(test_csv, args.attr, transform=None, mode='test')
            eval_data_loader = load_dataset(test_dataset, batch_size=args.batch_size, shuffle=False)
            evaluate(eval_data_loader, net, test_csv)
    if args.task == 'xray':
        test_csv_dir = './data/chestXray/csv/chexpert_processed_gender.csv'
        test_csv = pd.read_csv(test_csv_dir, low_memory=False).reset_index(drop=True)
        test_dataset = get_dataset(test_csv, args.attr, transform=None, mode='test')
        eval_data_loader = load_dataset(test_dataset, batch_size=args.batch_size, shuffle=False)
        evaluate(eval_data_loader, net, test_csv)

if __name__ == "__main__":
    main()