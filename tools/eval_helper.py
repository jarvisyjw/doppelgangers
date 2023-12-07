from . import io, matcher_helper

from multipledispatch import dispatch
from pathlib import Path
from vpr_val import logger
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay


# max recall @ 100% precision
def max_recall(precision: np.ndarray, recall: np.ndarray):
    recall_idx = np.argmax(recall)
    idx = np.where(precision == 1.0)
    max_recall = np.max(recall[idx])
    logger.debug(f'max recall{max_recall}')
    recall_idx = np.array(np.where(recall == max_recall)).reshape(-1)
    recall_idx = recall_idx[precision[recall_idx]==1.0]
    logger.debug(f'recall_idx {recall_idx}')
    r_precision, r_recall = float(np.squeeze(precision[recall_idx])), float(np.squeeze(recall[recall_idx]))
    logger.debug(f'precision: {r_precision}, recall: {r_recall}')
    return r_precision, r_recall

def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, average_precision, dataset = 'robotcar', exp_name = 'test'):
    # calculate max f2 score, and max recall
    f_score = 2 * (precision * recall) / (precision + recall)
    f_idx = np.argmax(f_score)
    f_precision, f_recall = np.squeeze(precision[f_idx]), np.squeeze(recall[f_idx])
    r_precision, r_recall = max_recall(precision, recall)
    plt.plot(recall, precision, label="{} (AP={:.3f})".format(dataset, average_precision))
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
    plt.title("Precision-Recall Curves for {}".format(exp_name))


# @dispatch(Path, str, str, str)
def evaluate(pointMap: Path, gt_file: str, dataset = 'SeqVal', exp_name = 'test'):
    with open(gt_file, 'r') as f:
        gt_labels = f.readlines()
    f.close()
    matches = []
    labels = []
    for gt_label in tqdm(gt_labels):
        query, reference, label = gt_label.strip('\n').split(', ')
        logger.debug(f'query: {query}, reference: {reference}, label: {label}')
        pointmap, inliers = matcher_helper.loadPointMap(query, reference, pointMap)
        # if inliers == None:
        #     logger.debug(f'original: {len(pointmap)}, inliers: {0}, label: {label}')
        #     matches.append(pointmap)
        #     labels.append(int(label))
        #     continue
        logger.debug(f'original: {len(pointmap)}, inliers: {len(inliers)}, label: {label}')
        matches.append(len(inliers))
        labels.append(int(label))
    matches_pts = np.array(matches)
    logger.debug(f'matches_pts: {matches_pts}')
    logger.debug(f'Max num of matches is {max(matches_pts)}')
    matches_pts_norm = matches_pts / max(matches_pts)
    average_precision = average_precision_score(labels, matches_pts_norm)
    precision, recall, TH = precision_recall_curve(labels, matches_pts_norm)
    plot_pr_curve(recall, precision, average_precision, dataset, exp_name)
    _, r_recall = max_recall(precision, recall)

    logger.info(f'\n' +
                f'Evaluation results: \n' +
                'Average Precision: {:.3f} \n'.format(average_precision) + 
                f'Maximum & Minimum matches: {np.max(matches_pts)}' + ' & ' + f'{np.min(matches_pts)} \n' +
                'Maximum Recall @ 100% Precision: {:.3f} \n'.format(r_recall))

