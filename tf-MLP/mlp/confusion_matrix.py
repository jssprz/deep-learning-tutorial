#!/usr/bin/env python
"""Provides the plot_confusion_matrix.
plot_confusion_matrix generates the confusion matrix from a lists of truths and predicted labels
"""

from textwrap import wrap
import re
import itertools
import tfplot
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='MyFigure/image', normalize=False):
    """
    :param predict_labels: These are you predicted classification categories
    :param correct_labels: These are your true classification categories.
    :param labels: This is a lit of labels which will be used to display the axis labels
    :param tensor_name: Name for the output summary tensor
    :param normalize: Set this to True if you want to normalize the values of the confusion matrix

    Returns:
        summary: TensorFlow summary

    Other items to note:
        - Depending on the number of category and the data, you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks don't line up due to rotations.
    """

    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], '.2f' if normalize else 'd') if cm[i, j] != 0 else '.',
                horizontalalignment='center', fontsize=6, verticalalignment='center', color='black')
    fig.set_tight_layout(True)

    return tfplot.figure.to_summary(fig, tag=tensor_name)