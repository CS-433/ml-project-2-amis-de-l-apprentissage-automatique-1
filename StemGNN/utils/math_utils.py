from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def evaluate(y, y_pred, num_classes):
    '''
    :param y: Ground truth labels, array of shape [batch_size].
    :param y_hat: Predictions, array of shape [batch_size, num_classes] (logits or probabilities).
    :param num_classes: Number of classes in the classification task.
    :return: Dictionary with F1 score and accuracy per class and overall.
    '''

    y = y.astype(int) - 1
    y_pred = y_pred.astype(int)

    # per-class F1 score and accuracy
    f1_per_class = f1_score(y, y_pred, labels=np.arange(num_classes), average=None)
    accuracy_per_class = []
    for cls in range(num_classes):
        mask = y == cls  # select instances belonging to this class
        if np.sum(mask) > 0:
            accuracy_per_class.append(np.mean(y_pred[mask] == cls))
        else:
            accuracy_per_class.append(0.0)  # handle case with no samples for the class

    # overall F1 score and accuracy
    f1_overall = f1_score(y, y_pred, average='macro')
    accuracy_overall = accuracy_score(y, y_pred)

    result = {
        'f1_per_class': f1_per_class.tolist(),
        'accuracy_per_class': accuracy_per_class,
        'f1_overall': f1_overall,
        'accuracy_overall': accuracy_overall
    }

    return result
