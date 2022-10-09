from dataclasses import dataclass

import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        print(f'Logistic Regression classifier of dim ({in_dim} {hid_dim} {out_dim})')

        self.nn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dim, bias=True),
        )

    def forward(self, x, return_feat=False):
        out = self.nn(x)
        if return_feat:
            return out, x
        return out


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        print(f'Logistic Regression classifier of dim ({in_dim} {hid_dim} {out_dim})')

        self.nn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, hid_dim / 2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dim, bias=True),
        )

    def forward(self, x):
        return self.nn(x)


class BertClassifier(nn.Module):
    FEAT_LEN = 768

    def __init__(self, raw_bert, classifier):
        super().__init__()
        self.bert = raw_bert
        self.fc = classifier

    def forward(self, x, return_feat=False):
        # x is a tokenized input
        # feature = self.bert(input_ids=x[0], token_type_ids=x[1], attention_mask=x[2])
        feature = self.bert(input_ids=x[0], attention_mask=x[2])
        # out = self.fc(feature.pooler_output.flatten(1))       # not good for our task     # (BS, E)
        out = self.fc(feature.last_hidden_state.flatten(1))  # (BS, T, E)
        if return_feat:
            return out, feature.last_hidden_state.flatten(1)
        return out


@dataclass
class BertClassiferHyperparams:
    mlp_size: int
    token_len: int
    embed_len: int


class SimpleEnsemble(nn.Module):
    """
    The simplest ensemble model, ie, averaging
    """

    def __init__(self, components):  # components is a list of models
        super(SimpleEnsemble, self).__init__()
        self.components = components

    def forward(self, inputs):
        assert len(self.components) == len(inputs)
        preds = []
        for model, input in zip(self.components, inputs):
            preds.append(model(input))
        return sum(preds) / len(preds)


class FixedWeightEnsemble(nn.Module):
    """
    Learn a fixed set of weights
    """

    def __init__(self, components):
        super(FixedWeightEnsemble, self).__init__()
        self.components = components
        self.weights = nn.Linear(1, len(components), bias=False)
        self.weightsInput = torch.tensor([1], dtype=torch.float).cuda()

    def forward(self, inputs):
        assert len(self.components) == len(inputs)

        preds = []
        for model, input in zip(self.components, inputs):
            pred = model(input)
            preds.append(pred)

        weights = self.weights(self.weightsInput)
        for i, weight in enumerate(weights):
            preds[i] = preds[i] * weight

        return sum(preds)


class DynamicWeightEnsemble(nn.Module):
    """
    Learn the dynamic weights for different components
    """

    def __init__(self, components, total_feat_len, dropout=0.2, hidden_len=256):
        super(DynamicWeightEnsemble, self).__init__()
        self.components = components
        self.attention = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_feat_len, hidden_len, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_len, len(components), bias=True),
            nn.Softmax()
        )

    def forward(self, inputs):
        assert len(self.components) == len(inputs)

        preds, feats = [], []
        for model, input in zip(self.components, inputs):
            pred, feat = model(input, return_feat=True)
            preds.append(pred)
            feats.append(feat)

        weights = self.attention(torch.cat(feats, dim=1))
        weights = torch.transpose(weights, 0, 1)
        for i in range(weights.size(0)):
            for j in range(weights.size(1)):
                preds[i][j] *= weights[i][j]

        return sum(preds)


class AggregateFeatEnsemble(nn.Module):
    """
    Learn the dynamic weights for different components
    """

    def __init__(self, components, total_feat_len, num_classes, dropout=0.2, hidden_len=256):
        super(AggregateFeatEnsemble, self).__init__()
        self.components = nn.ModuleList(components)
        self.nn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_feat_len, hidden_len, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_len, num_classes, bias=True)
        )
        #         self.nn2 = nn.Sequential(
        #             nn.Dropout(dropout),
        #             nn.Linear(total_feat_len, hidden_len, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #             nn.Dropout(dropout),
        #             nn.Linear(hidden_len, num_classes, bias=True)
        #         )
        print(f'aggregate feat ensemble, input feat len {total_feat_len}, hidden size {hidden_len}')

    def forward(self, inputs, return_feats=False, return_preds=False):
        assert len(self.components) == len(inputs)

        preds, feats = [], []
        for model, input in zip(self.components, inputs):
            pred, feat = model(input, return_feat=True)
            preds.append(pred)
            feats.append(feat)

        #         hidden_feat = self.nn(torch.cat(feats, dim=1))
        #         pred = self.nn2(hidden_feat)
        pred = self.nn(torch.cat(feats, dim=1))

        out = [pred]
        if return_feats:
            out.append(feats)
        if return_preds:
            out.append(preds)
        if len(out) == 1:
            return out[0]
        else:
            return out

    # def forward(self, feats):
    #     return self.nn(feats)


class EnsembleClassifier(nn.Module):
    FEAT_LEN = 768

    def __init__(self, raw_bert, styleClassifier, charClassifier, bertClassifier, finalClassifier):
        super().__init__()
        self.bert = raw_bert
        self.styleClassifier = styleClassifier
        self.charClassifier = charClassifier
        self.bertClassifier = bertClassifier
        self.finalClassifier = finalClassifier

    def forward(self, x, return_feat=False):
        # x is a tokenized input
        # print("ENS Forward")

        stylePred = self.styleClassifier(x[0])

        charPred = self.charClassifier(x[1])

        bertFeature = self.bert(x[2], x[3]).last_hidden_state.flatten(1)
        bertPred = self.bertClassifier(bertFeature)
        # print(stylePred.shape)
        # print(charPred.shape)
        # print(bertFeature.shape)
        # print(bertPred.shape)
        # print(x[0].shape)
        # print(x[1].shape)
        ensembleTensor = torch.cat((stylePred, charPred, bertPred, x[0], x[1], bertFeature), dim=1)
        # out = self.fc(feature.pooler_output.flatten(1))
        out = self.finalClassifier(ensembleTensor)
        if return_feat:
            return out, bertFeature
        return out
