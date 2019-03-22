import matplotlib.pyplot as plt
import numpy as np
import itertools


# todo make sure that classes and hypnogram have same stage mode
def plot(samples_indices, samples_values, classes, hypnogram):
    c = ['b' if c == h else 'r' for c, h in zip(classes, hypnogram)]
    plt.scatter(samples_indices, samples_values, c=c, s=0.05)
    plt.show()


def _prepare_plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()
    # plt.savefig(file_path)
    return plt


def show_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    s_plt = _prepare_plot_confusion_matrix(cm, classes, normalize, title, cmap)
    s_plt.show()
    return s_plt


def save_to_file_confusion_matrix(cm, classes, file_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    f_plt = _prepare_plot_confusion_matrix(cm, classes, normalize, title, cmap)
    f_plt.savefig(file_path)
    return f_plt
