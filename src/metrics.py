# Calculate accuracy from two arrays
from sklearn.metrics import precision_score, accuracy_score,recall_score,balanced_accuracy_score
from sklearn.metrics import f1_score
def Metrics(targets,preds,threshold = 0.5):
    '''
    targets: numpy array from 0 to n_classes-1
    preds  : numpy array from 0 to n_classes -1
    '''
    preds = (preds>threshold)*1
    # Calculate Accuracy
    recall    = recall_score(targets,preds, average='micro')
    precision = precision_score(targets, preds, average='micro')
    bal_acc   = balanced_accuracy_score(targets, preds)
    acc       = accuracy_score(targets, preds)
    f1  = f1_score(targets,preds, average='micro') 
    return {'recall': recall,
            'precision':precision,
            'balanced_acc': bal_acc,
            'accuracy'    : acc,
            'f1_score'    : f1}
