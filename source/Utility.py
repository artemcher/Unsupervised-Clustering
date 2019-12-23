import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# Module with miscellaneous utility functions.

def calc_accuracy(y_true, y_pred):
    '''
    Calculate clustering accuracy.
    Args:
        y_true: true labels, numpy.array with shape '(n_samples,)'
        y_pred: predicted labels, numpy.array with shape '(n_samples,)''
    Returns clustering accuracy, in range [0,1].
    '''
    y_true = y_true.astype("int")
    assert y_pred.size == y_true.size

    # make and fill confusion matrix
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # solve linear sum assignment problem and calculate accuracy
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = np.sum(w[row_ind, col_ind])*1.0 / y_pred.size
    return acc, col_ind


def test_accuracy():
    '''
    Test calculation of clustering accuracy with trivial values.
    '''
    arr_len = 27
    print("Generating two random binary arrays of size {}".format(arr_len))
    dummy_true = np.random.randint(2, size=(arr_len))
    dummy_pred = np.random.randint(2, size=(arr_len))

    print("True  array:", dummy_true)
    print("Pred. array:", dummy_pred)

    mismatches = np.sum(np.abs(dummy_true-dummy_pred))
    matches = arr_len - mismatches
    print("Matches: {} out of {}.".format(matches, arr_len))
    print("Mismatches: {} out of {}.".format(mismatches, arr_len))

    expected_acc = 0
    if matches >= mismatches:
        print("Majority matches. No remapping required.")
        expected_acc = matches/arr_len
        print("Expected accuracy (matches/array_size):", expected_acc)
    else:
        print("Majority mismatches. Remapping required: 0 -> 1, 1 -> 0.")
        expected_acc = mismatches/arr_len
        print("Expected accuracy (mismatches/array_size):", expected_acc)

    actual_acc = calc_accuracy(dummy_true, dummy_pred)[0]
    print("Actual accuracy:", actual_acc)
    print("Total Error:", np.abs(expected_acc - actual_acc))



if __name__=="__main__":
    # EXAMPLE: test accuracy calculation
    test_accuracy()