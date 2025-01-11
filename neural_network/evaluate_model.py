import torch
from neural_network import Softmax


def evaluate_model(model, test_loader):

    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        S = model.forward(data)

        softmax = Softmax()

        softmax_probs = softmax.forward(S)

        y_pred = torch.argmax(softmax_probs, dim=0)

        accuracy = calc_accuracy(y_pred, target)
        accuracy_sum += accuracy
        precision = macro_precision(y_pred, target)
        precision_sum += precision
        recall = macro_recall(y_pred, target)
        recall_sum += recall
        f1 = macro_f1(y_pred, target)
        f1_sum += f1

    accuracy_mean = accuracy_sum / len(test_loader)
    precision_mean = precision_sum / len(test_loader)
    recall_mean = recall_sum / len(test_loader)
    f1_mean = f1_sum / len(test_loader)
    print(
        f"accuracy: {accuracy_mean}, precision: {precision_mean}, recall: {recall_mean}, f1: {f1_mean}"
    )


def calc_accuracy(y_pred, y_true):
    """
    Calculate accuracy for multi-class classification using PyTorch tensors.
    """
    correct_predictions = (
        (y_pred == y_true).sum().item()
    )  # sum the number of correct predictions
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def macro_precision(y_pred, y_true):

    # find the number of classes
    num_classes = len(torch.unique(y_true))

    # initialize precision to 0
    precision = 0

    # loop over all classes
    for class_ in list(y_true.unique()):

        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_pred, temp_true)

        # compute false positive for current class
        fp = false_positive(temp_pred, temp_true)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision

    # calculate and return average precision over all classes
    precision /= num_classes

    return precision


def macro_recall(y_pred, y_true):

    # find the number of classes
    num_classes = len(torch.unique(y_true))

    # initialize recall to 0
    recall = 0

    # loop over all classes
    for class_ in list(y_true.unique()):

        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_pred, temp_true)

        # compute false negative for current class
        fn = false_negative(temp_pred, temp_true)

        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # keep adding recall for all classes
        recall += temp_recall

    # calculate and return average recall over all classes
    recall /= num_classes

    return recall


def macro_f1(y_pred, y_true):

    # find the number of classes
    num_classes = len(torch.unique(y_true))

    # initialize f1 to 0
    f1 = 0

    # loop over all classes
    for class_ in list(y_true.unique()):

        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_pred, temp_true)

        # compute false negative for current class
        fn = false_negative(temp_pred, temp_true)

        # compute false positive for current class
        fp = false_positive(temp_pred, temp_true)

        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)

        temp_f1 = (
            2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)
        )

        # keep adding f1 score for all classes
        f1 += temp_f1

    # calculate and return average f1 score over all classes
    f1 /= num_classes

    return f1


def true_positive(y_pred, y_true):

    tp = 0

    for yt, yp in zip(y_pred, y_true):

        if yt == 1 and yp == 1:
            tp += 1

    return tp


def true_negative(y_pred, y_true):

    tn = 0

    for yt, yp in zip(y_pred, y_true):

        if yt == 0 and yp == 0:
            tn += 1

    return tn


def false_positive(y_pred, y_true):

    fp = 0

    for yt, yp in zip(y_pred, y_true):

        if yt == 0 and yp == 1:
            fp += 1

    return fp


def false_negative(y_pred, y_true):

    fn = 0

    for yt, yp in zip(y_pred, y_true):

        if yt == 1 and yp == 0:
            fn += 1

    return fn
