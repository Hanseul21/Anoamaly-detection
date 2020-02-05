from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from torch.distributions import Normal
import matplotlib.pyplot as plt
import sys, os
import torch
import numpy as np
import pandas as pd

class evaluate():
    def __init__(self, data_name, is_anomaly=1):
        self.output = []
        self.is_anomaly = is_anomaly
        if data_name == 'kpi':
            self.root = os.path.join('record', 'KPI')
        elif data_name == 'yahoo':
            self.root = os.path.join('record', 'yahoo')
        elif data_name == 'nab':
            self.root = os.path.join('record', 'NAB')
        else:
            print('Evaluation/evaluate')
            print('there is no ',data_name)
            sys.exit()

    def f1(self, y_pred, y_true):
        if type(y_pred).__module__ == torch.__name__:
            y_pred = y_pred.data.cpu().numpy().reshape(-1)
        if type(y_true).__module__ == torch.__name__:
            y_true = y_true.data.cpu().numpy().reshape(-1)

        return f1_score(y_true, y_pred)

    def roc_auc(self, y_raw, y_true, title='', plot=False):
        if type(y_raw).__module__ == torch.__name__:
            y_raw = y_raw.data.cpu().numpy().reshape(-1)
        if type(y_true).__module__ == torch.__name__:
            y_true = y_true.data.cpu().numpy().reshape(-1)

        fpr, tpr, threshold = roc_curve(y_true, y_raw)
        ROCAUC = auc(fpr, tpr)

        if plot:
            plt.plot(fpr, tpr,label='ROC-curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate - recall')
            plt.title(title)
            plt.show()
        return ROCAUC, fpr, tpr

    def pr_auc(self, y_raw, y_true, title='', plot=False):
        if type(y_raw).__module__ == torch.__name__:
            y_raw = y_raw.data.cpu().numpy().reshape(-1)
        if type(y_true).__module__ == torch.__name__:
            y_true = y_true.data.cpu().numpy().reshape(-1)

        precision, recall, thresholds = precision_recall_curve(y_true, y_raw, pos_label=self.is_anomaly)
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        f1 = 2 *precision * recall / (precision + recall)
        f1 = np.nan_to_num(f1)
        PRAUC = auc(recall, precision)
        if plot:
            plt.plot(recall, precision,label='PR-curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.show()
        return PRAUC, precision, recall, thresholds, f1

    def anomaly_seg(self, y_pred, y_true, delay=100):
        if type(y_pred).__module__ == torch.__name__:
            y_pred = y_pred.data.cpu().numpy().reshape(-1)
        if type(y_true).__module__ == torch.__name__:
            y_true = y_true.data.cpu().numpy().reshape(-1)

        splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
        is_anomaly = y_true[0] == 1
        new_predict = np.array(y_pred)
        pos = 0

        for sp in splits:
            if is_anomaly:
                if 1 in y_pred[pos:min(pos + delay + 1, sp)]:
                    new_predict[pos: sp] = 1
                else:
                    new_predict[pos: sp] = 0
            is_anomaly = not is_anomaly
            pos = sp
        sp = len(y_true)

        if is_anomaly:  # anomaly in the end
            if 1 in y_pred[pos: min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0

        return new_predict

    def get_log_prob(self, x, mu=None, std=None):
        """
        negative log likelihood
        more less the value, higher the probability of normal
        """
        if mu is None:
            mu = torch.ones_like(x)
            std = torch.zeros_like(x)
        return -(Normal(mu, std).log_prob(x))

    def get_score(self, y_raw, threshold, type='recon_prob'):
        if type == 'recon_prob':
            y_pred = torch.where(y_raw > threshold, torch.ones_like(y_raw), torch.zeros_like(y_raw))
        else:
            print('there is no', type)
            sys.exit()
        return y_pred

    def get_result(self, y_raw, y_true, writer=None, idx=None, plot=False):
        true = (y_true != 0).sum(dim=1).sum(dim=0).item()
        if true == 0:
            prec, recall, f1, f1_corr = [0],[0],[0],0
            argmax = 0
        else:
            _, prec, recall, th, f1 = self.pr_auc(y_raw, y_true, plot=plot)
            argmax = np.argmax(f1)
            y_pred = self.get_score(y_raw, th[argmax])
            new_pred = self.anomaly_seg(y_pred, y_true)
            # f1 = evaluator.f1(y_pred, y_true)
            f1_corr = self.f1(new_pred, y_true)

        self.true, self.prec, self.recall, self.bestf1, self.f1_corr = true, recall[argmax], prec[argmax], np.max(f1), f1_corr
        print('true {0} precision {1:.5f} recall {2:.5f} Best f1-score {3:.5f} corrected f1-score {4:.5f}'
              .format(self.true, self.prec, self.recall, self.bestf1, self.f1_corr))
        if writer != None:
            writer.add_scalars('performance', {'true': self.true,
                                               'precision': self.prec,
                                               'recall': self.recall,
                                               'f1': self.bestf1,
                                               'corrected_f1': self.f1_corr}, idx)

        return self.true, self.prec, self.recall, self.bestf1, self.f1_corr

    def record(self, idx):
        self.output.append([idx, self.true, self.prec, self.recall, self.bestf1, self.f1_corr])

    def get_record(self, title):
        df = pd.DataFrame(self.output, columns=['idx', 'true', 'precision', 'recall', 'f1score', 'threshold'])
        df.to_csv(os.path.join(self.root, title + '.csv'), float_format='%.4f')
        return os.path.join(self.root, title + '.csv')

    # def anomaly_seg(self, y_pred, y_true, delay=1, is_anomaly=1):
    #     """
    #     modify prediction using anomaly segmentation
    #
    #     :param y_pred: Tensor. prediction
    #     :param y_true: Tensor. Ground Truth
    #     :param delay: Scalar. allowed delay
    #     :param is_anomaly: Scalar. value of anomaly class
    #     :return: new_pred. Tensor. modified prediction
    #     """
    #     idx_fin = 0
    #     new_pred = y_pred
    #     for seq in y_true:
    #         for idx in range(len(seq)):
    #             if seq[idx] == is_anomaly:
    #                 # anomaly segment
    #                 for i, ano_seg in enumerate(seq[idx:]):
    #                     if ano_seg != is_anomaly:
    #                         idx_fin = idx + i
    #                         break
    #                 if idx_fin > delay:
    #                     tilde = delay
    #                 else:
    #                     tilde = idx_fin
    #                 if (seq[idx:tilde] == is_anomaly).sum() >0:
    #                     new_pred[idx:idx_fin] = 1
    #                 else:
    #                     new_pred[idx:idx_fin] = 0
    #
    #     return new_pred
