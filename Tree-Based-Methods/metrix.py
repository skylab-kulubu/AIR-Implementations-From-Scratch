import numpy as np
import pandas as pd

def confusion_matrix(actual, predicted, classes=[0, 1]):
    predicted_pos = (predicted == classes[1])
    predicted_neg = (predicted == classes[0])

    actual_pos = (actual == classes[1])
    actual_neg = (actual == classes[0])

    true_pos = actual[ (actual_pos) & (predicted_pos) ]
    false_pos = actual[ (actual_neg) & (predicted_pos) ]

    false_neg = actual[ (actual_pos) & (predicted_neg) ]
    true_neg = actual[ (actual_neg) & (predicted_neg) ]

    conf_mat = np.array([
        [ true_pos.size, false_pos.size ],
        [ false_neg.size, true_neg.size ]
    ])

    return pd.DataFrame(conf_mat, columns = ['Actual Positive', 'Actual Negative'], 
                                    index = ['Predicted Positive', 'Predicted Negative'])

def accuracy(actual, predicted):
    return actual [ actual == predicted ].size / actual.size

def precision(actual, predicted, classes=[0, 1]):
    predicted_pos = (predicted == classes[1])
    actual_pos = (actual == classes[1])

    true_pos = actual[ (actual_pos) & (predicted_pos) ]
    
    return true_pos.size / predicted_pos[predicted_pos].size
    
def recall(actual, predicted, classes=[0, 1]):
    predicted_pos = (predicted == classes[1])
    actual_pos = (actual == classes[1])

    true_pos = actual[ (actual_pos) & (predicted_pos) ]
    
    return true_pos.size / actual[actual_pos].size

def f1_score(actual, predicted, classes=[0, 1]):
    prec = precision(actual, predicted, classes=classes)
    rec = recall(actual, predicted, classes=classes)

    return 2 * prec * rec / (prec + rec)