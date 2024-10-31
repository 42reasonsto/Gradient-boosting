import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import random
import pandas as pd

import os

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

train_values = pd.read_csv("\\train_values.csv")
train_targets_counted = pd.read_csv("\\train_targets_counted.csv")
train_targets_noncounted = pd.read_csv("\\train_targets_noncounted.csv")

test_values = pd.read_csv("\\test_values.csv")
sample_submission = pd.read_csv("\\sample_submission.csv")

train_drug = pd.read_csv("\\train_drug.csv")

params = {"number_gens_pca": 50,
          "number_living_cells_pca": 20,
          "batch_size": 256,
          "lr": 1e-3,
          "weight_decay": 1e-5,
          "number_folds": 5,
          "early_stopping_steps": 5,
          "hidden_size": 512,
          "boost_rate": 1.0,
          "number_nets": 20,
          "epochs_per_stage": 1,
          "correct_epoch": 1,
          "model_order": "second"
          }

GENS = [col for col in train_values.columns if col.startswith('g-')]
LIVING_CELLS = [col for col in train_values.columns if col.startswith('c-')]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)

# GENS
number_components = params["number_gens_pca"]

data = pd.concat([pd.DataFrame(train_values[GENS]), pd.DataFrame(test_values[GENS])])
new_data = (PCA(n_components=number_components, random_state=42).fit_transform(data[GENS]))
new_train_values = new_data[:train_values.shape[0]];
new_test_values = new_data[-test_values.shape[0]:]

new_train_values = pd.DataFrame(new_train_values, columns=[f'pca_G-{i}' for i in range(number_components)])
new_test_values = pd.DataFrame(new_test_values, columns=[f'pca_G-{i}' for i in range(number_components)])

train_values = pd.concat((train_values, new_train_values), axis=1)
test_values = pd.concat((test_values, new_test_values), axis=1)

# LIVING_CELLS
number_components = params["number_living_cells_pca"]

data = pd.concat([pd.DataFrame(train_values[LIVING_CELLS]), pd.DataFrame(test_values[LIVING_CELLS])])
new_data = (PCA(n_components=number_components, random_state=42).fit_transform(data[LIVING_CELLS]))
new_train_values = new_data[:train_values.shape[0]];
new_test_values = new_data[-test_values.shape[0]:]

new_train_values = pd.DataFrame(new_train_values, columns=[f'pca_C-{i}' for i in range(number_components)])
new_test_values = pd.DataFrame(new_test_values, columns=[f'pca_C-{i}' for i in range(number_components)])

train_values = pd.concat((train_values, new_train_values), axis=1)
test_values = pd.concat((test_values, new_test_values), axis=1)

data = pd.concat([pd.DataFrame(train_values), pd.DataFrame(test_values)])
new_data = pd.get_dummies(data, columns=["cp_time", "cp_dose"])

new_train_values = new_data[:train_values.shape[0]];
new_test_values = new_data[-test_values.shape[0]:]

train_values = new_train_values
test_values = new_test_values
train_values

value_cols = train_values.columns[4:].tolist()
params["feat_d"] = len(value_cols)
train_values

value_cols[:10]

train = train_values.merge(train_targets_counted, on='sig_id')
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
test = test_values[test_values['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_counted.columns]

train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

train

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

##CV STRATEGY

folds_ = train.copy()

folds = []

##LOAD FILES
train_feats = train_values
scored = target
drug = train_drug
drug = drug.loc[train_feats['cp_type'] == 'trt_cp', :]
targets = target_cols
scored = scored.merge(drug, on='sig_id', how='left')

##LOCATE DRUGS
vc = scored.drug_id.value_counts()
vc1 = vc.loc[vc <= 18].index.sort_values()
vc2 = vc.loc[vc > 18].index.sort_values()

##STRATIFY DRUGS 18X OR LESS
diction1 = {};
diction2 = {}
k_validation = MultilabelStratifiedKFold(n_splits=params["number_folds"], shuffle=True, random_state=0)
temp = scored.groupby('drug_id')[targets].mean().loc[vc1]
for fold, (idxT, idxV) in enumerate(k_validation.split(temp, temp[targets])):
    dd = {k: fold for k in temp.index[idxV].values}
    diction1.update(dd)

##STRATIFY DRUGS MORE THAN 18X
k_validation = MultilabelStratifiedKFold(n_splits=params["number_folds"], shuffle=True, random_state=0)
temp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
for fold, (idxT, idxV) in enumerate(k_validation.split(temp, temp[targets])):
    dd = {k: fold for k in temp.sig_id[idxV].values}
    diction2.update(dd)

##ASSIGN FOLDS
scored['fold'] = scored.drug_id.map(diction1)
scored.loc[scored.fold.isna(), 'fold'] = \
scored.loc[scored.fold.isna(), 'sig_id'].map(diction2)
scored.fold = scored.fold.astype('int8')
folds.append(scored.fold.values)

del scored['fold']

s = np.stack(folds)
train["kfold"] = s.reshape(-1, )

train

class MyDataset:
    def __init__(self, values, targets):
        self.values = values
        self.targets = targets

    def __len__(self):
        return (self.values.shape[0])

    def __getitem__(self, idx):
        diction = {
            'x': torch.tensor(self.values[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return diction


##Dataset Classes
class TestDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return (self.values.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.values[idx, :], dtype=torch.float)
        }
        return dct


##Dynamic Model
from enum import Enum


class Type(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3


class DynamicModelNet(object):
    def __init__(self, c0, lr):
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True, device="cuda"))

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)


    def forward(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0 = np.repeat(self.c0.detach().cpu().numpy().reshape(1, -1), batch, axis=0)
            return None, torch.Tensor(c0).cuda()
        middle_feat_cumulat = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cumulat is None:
                    middle_feat_cumulat, prediction = m(x, middle_feat_cumulat)
                else:
                    middle_feat_cumulat, pred = m(x, middle_feat_cumulat)
                    prediction += pred
        return middle_feat_cumulat, self.c0 + self.boost_rate * prediction


    def forward_grad(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0 = np.repeat(self.c0.detach().cpu().numpy().reshape(1, -1), batch, axis=0)
            return None, torch.Tensor(c0).cuda()
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction


    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicModelNet(d['c0'], d['lr'])
        net.boost_rate = d['boost_rate']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net


    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate}
        torch.save(d, path)


##Weak Models

class MLP_1HidLay(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_1HidLay, self).init()
        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_in, dim_hidden1),
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_hidden1, 206))
        if bn:
            self.bn = nn.BatchNorm1d(dim_hidden1)
            self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.layer1(x)
        return out, self.layer2(out)

    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        model = MLP_1HidLay(dim_in, params["hidden_size"], params["hidden_size"])
        return model


class MLP_2HidLay(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_2HidLay, self).init()

        self.bn2 = nn.BatchNorm1d(dim_in)

        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_in, dim_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden1),
            nn.Dropout(0.4),
            nn.Linear(dim_hidden1, dim_hidden2)
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_hidden2, 206)
        )


    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        middle_feat = self.layer1(x)
        out = self.layer2(middle_feat)
        return middle_feat, out


    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        model = MLP_2HidLay(dim_in, params["hidden_size"], params["hidden_size"])
        return model


from torch.nn.modules.loss import _WeightedLoss


class SmoothingBCEwithLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().init(smoothing=0.001, weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothingBCEwithLogits._smooth(targets, inputs.size(-1),
                                                 self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def get_optim(params, lr, weight_decay):
    optimizer = optim.Adam(params, lr, weight_decay=weight_decay)  # SGD
    return optimizer


def log_loss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss()  # Binary cross entopy loss
    for data in test_loader:
        x = data["x"].cuda()
        y = data["y"].cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        loss += loss_f(out, y)
        total += 1

    return loss / total


device = "cuda" if torch.cuda.is_available() else "cpu"

