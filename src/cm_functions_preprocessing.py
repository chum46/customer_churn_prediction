import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

def ss_scale(self, X_train, X_test): 
    """
    Feature scaling using sklearn StandardScaler
    note: StandardScaler cannot guarantee balanced feature scales in the presence of outliers.
    Uses scikit-learn: https://scikit-learn.org/stable/index.html
    ...

    Parameters
    ----------
    pandas DataFrame or Series
        Train and Test feature datasets

    Returns
    -------
    pandas DataFrame or Series
        Scaled train and test features

    """
    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    return X_train_scaled, X_test_scaled
