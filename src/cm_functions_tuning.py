from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, recall_score
from sklearn.metrics import mean_squared_error, auc, average_precision_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def LRM_regularization_tune(X_train_sc, y_train):
    """
    {In progress}
    ...

    Parameters
    ----------

    Returns
    -------

    Example
    --------

    """
    C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    names = [0.001, 0.01, 0.1, 1, 10, 100]
    colors = sns.color_palette('Set2')

    plt.figure(figsize=(10, 8))

    for n, c in enumerate(C_param_range):
        # Fit a model
        logreg = LogisticRegression(C=c, solver='liblinear')
        model_log = logreg.fit(X_train_sc, y_train)
        print(model_log) # Preview model params

        # Predict
        y_hat_train = logreg.predict(X_train_sc)

        y_train_score = model_log.decision_function(X_train_sc)

        fpr, tpr, thresholds = roc_curve(y_train, y_train_score)

        print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
        print('-------------------------------------------------------')
        lw = 2
        plt.plot(fpr, tpr, color=colors[n],
                 lw=lw, label='ROC curve Normalization Weight: {}'.format(names[n]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show();
    return 
    

def crossValidate(): 
    """
     
    
    ...

    Attributes
    ----------


    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
        
    """
    return
        