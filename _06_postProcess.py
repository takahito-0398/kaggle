import numpy as np
import keras
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools

from _00_utils import myroc

# from _00_utils import get_batch



# LOAD SUBMISSION
FUDGE = 1.0

p_target = "./pred_by_traindata/pred_train_ResNet50V2_type5_3epochs.csv"
df = pd.read_csv(p_target)
print(df.head())
y_pred = df.target.values
y_true = df.true_label.values
print(y_pred)
print(y_true)
auc = myroc(y_true, y_pred)


print(auc)
# cm_val = confusion_matrix(y_true, y_pred)
# tn, fp, fn, tp = cm_val.flatten()
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1 = 2 * precision * recall / (precision + recall)
raise
# CONVERT PROBS TO ODDS, APPLY MULTIPLIER, CONVERT BACK TO PROBS
def scaler(probs, factors):
    a = factors["a"]
    b = factors["b"]
    c = factors["c"]
    probs = probs.copy()
    idx = np.where(probs!=1)[0]
    odds = probs[idx] / (1-probs[idx])
    odds = a * (odds ** b) + c
    probs[idx] =  odds/(1+odds)
    return probs


def scaling(df):
    y_pred = df.target.values
    y_true = df.true_label.value
    a = [10**i for i in range(10)]
    b = [i for i in range(-5,5)]
    c = [10**i for i in range(10)]
    print(a,b,c)
    iter_X = itertools.product(a, b, c)
    for x in iter_x:
        print(x)



# TRAIN AND TEST MEANS
mean_pred = df.iloc[:,1].mean()
print(mean_pred)
s = (mean_pred/(1-mean_pred))/FUDGE
df.iloc[:,1] = scaler(df.iloc[:,1].values,s)

print(df.head())
FILE_save = "./submission/submission_ResNet50V2_type5_3epochs_pp.csv"
df.to_csv(FILE_save,index=False)