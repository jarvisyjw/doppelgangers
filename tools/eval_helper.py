from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from scipy.special import softmax
import argparse


# max recall @ 100% precision
def max_recall(precision: np.ndarray, recall: np.ndarray):
    recall_idx = np.argmax(recall)
    idx = np.where(precision == 1.0)
    max_recall = np.max(recall[idx])
    recall_idx = np.array(np.where(recall == max_recall)).reshape(-1)
    recall_idx = recall_idx[precision[recall_idx]==1.0]
    r_precision, r_recall = float(np.squeeze(precision[recall_idx])), float(np.squeeze(recall[recall_idx]))
    return r_precision, r_recall

def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, average_precision):
    # calculate max f2 score, and max recall
    f_score = 2 * (precision * recall) / (precision + recall)
    f_idx = np.argmax(f_score)
    f_precision, f_recall = np.squeeze(precision[f_idx]), np.squeeze(recall[f_idx])
    r_precision, r_recall = max_recall(precision, recall)
    plt.plot(recall, precision, label="(AP={:.5f}, MR={:.5f})".format(average_precision, r_recall))
    plt.vlines(r_recall, np.min(precision), r_precision, colors='green', linestyles='dashed')
    plt.vlines(f_recall, np.min(precision), f_precision, colors='red', linestyles='dashed')
    plt.hlines(f_precision, np.min(recall), f_recall, colors='red', linestyles='dashed')
    plt.hlines(r_precision, np.min(recall), r_recall, colors='green', linestyles='dashed')
    plt.xlim(0, None)
    plt.ylim(np.min(precision), None)
    plt.text(f_recall + 0.005, f_precision + 0.005, '[{:.3f}, {:.3f}]'.format(f_recall, f_precision))
    plt.text(r_recall + 0.005, r_precision + 0.005, '[{:.3f}, {:.3f}]'.format(r_recall, r_precision))
    plt.scatter(f_recall, f_precision, marker='o', color='red', label='Max F score')
    plt.scatter(r_recall, r_precision, marker='o', color='green', label='Max Recall')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curves")


def parser():
    parser = argparse.ArgumentParser(description='plot pr curve')
    parser.add_argument('--val_log', type=str, help='validation log file')
    parser.add_argument('--pr', action='store_true', help='plot pr curve')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser()
    if args.pr:
        print("Plotting and Saving PR Curve")
        val_log = Path(args.val_log, 'test_doppelgangers_list.npy')
        # read result list
        result_list = np.load(val_log, allow_pickle=True)
        result_list = result_list.tolist()
        # process result
        pred = result_list['pred']
        gt_list = result_list['gt']
        prob = result_list['prob']
        scores = softmax(prob, axis=1)[:, 1]
        # plot PR Curve
        precision, recall, TH = precision_recall_curve(gt_list, scores)
        average_precision = average_precision_score(gt_list, scores)
        plot_pr_curve(recall, precision, average_precision)
        # save figure
        plt.savefig(Path(args.val_log,'pr_curve.pdf'))
