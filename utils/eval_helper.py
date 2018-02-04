
import numpy as np
from sklearn.metrics import roc_curve, auc

def get_roc_auc(yval, ypred, classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_list = []
    fpr_list=[]
    tpr_list=[]
    for i in range(len(list_classes)):
        fpr[i], tpr[i], _ = roc_curve(yval[:,i], y_val_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        print ("{:15s} ROC AUC = {:.5f}".format(list_classes[i], roc_auc[i]))

        roc_auc_list.append(roc_auc[i])
        fpr_list.append(fpr[i])
        tpr_list.append(tpr[i])

    print ("")
    print ("{:15s} ROC AUC = {:.5f}".format('Average', np.mean(roc_auc_list)))

    return tpr_list, fpr_list, roc_auc_list
