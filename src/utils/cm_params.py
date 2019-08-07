def accuracy(tp, fp, fn, tn):
    """

    :param tp: true positive
    :param fp: false positive
    :param fn: false negative
    :param tn: true negative
    :return: accuracy
    """
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else -1


def sensitivity(tp, fn):
    """

    :param tp: true positive
    :param fn: false negative
    :return: sensitivity
    """
    return tp / (tp + fn) if (tp + fn) != 0 else -1


def specificity(fp, tn):
    """

    :param fp: false positive
    :param tn: true negative
    :return: specificity
    """
    return tn / (tn + fp) if (tn + fp) != 0 else -1


def positive_predictive_value(tp, fp):
    """

    :param tp: true positive
    :param fp: false positive
    :return: positive_predictive_value
    """
    return tp / (tp + fp) if (tp + fp) != 0 else -1


def negative_predictive_value(fn, tn):
    """

    :param fn: false negative
    :param tn: true negative
    :return: negative_predictive_value
    """
    return tn / (tn + fn) if (tn + fn) != 0 else -1
