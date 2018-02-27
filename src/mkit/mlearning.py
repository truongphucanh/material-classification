import numpy
from mkit import util

def calculate_accuracy(y, y_pred):
    """Caculate accuracy for a testing
    
    Arguments:
        y {array} -- (n_samples) expected labels
        y_pred {array} -- (n_samples) predicted labels
    
    Returns:
        float -- accuracy
    """
    mis_indices = np.where(y != y_pred)[0]
    accuracy = 1.0 - ( 1.0 * len(mis_indices) / len(y))
    return accuracy

def get_labels(probs, classes):
    """Get label from probability results.
    
    Arguments:
        probs {array} -- (n_samples, n_classes) List of probability results from model.predict_proba(X)
        classes {array} -- (n_classes) List of classes from model.classes_ (trained model)
    """
    labels = []
    for prob in probs:
        i = util.index_max(prob)
        labels.append(classes[i])
    return numpy.array(labels)
