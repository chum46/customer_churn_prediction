import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample

def ss_scale(X_train, X_test): 
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

def mm_scale(X_train, X_test): 
    """
    Feature scaling using sklearn MinMaxScaler
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
    mm = MinMaxScaler()

    X_train_scaled = mm.fit_transform(X_train)
    X_test_scaled = mm.transform(X_test)
    return X_train_scaled, X_test_scaled

def prepare_dataset (df):
#     state_col = df.pop("state")
    df = df.drop("area_code", axis=1)
    df.voice_mail_plan.replace((True, False), (1, 0), inplace = True)
    df.international_plan.replace((True, False), (1, 0), inplace = True)
    df.churn.replace((True, False), (1, 0), inplace = True)
    y = df.churn
    X = df.drop("churn", axis=1)
    
    X2 = df.drop("churn", axis=1)
    X2['total_charge'] = df.total_day_charge+df.total_eve_charge+df.total_night_charge+df.total_intl_charge
    
    X2['avg_charge_per_day'] =(df.total_day_charge+df.total_eve_charge+
                            df.total_night_charge+df.total_intl_charge)/df.account_length
    X2['total_minutes'] = df.total_day_minutes+df.total_eve_minutes+df.total_night_minutes+df.total_intl_minutes
    X2['avg_min_per_day'] =(df.total_day_minutes+df.total_eve_minutes+
                            df.total_night_minutes+df.total_intl_minutes)/df.account_length
    X2['total_calls'] = df.total_day_calls+df.total_eve_calls+df.total_night_calls+df.total_intl_calls
    X2['avg_calls_per_day'] =(df.total_day_calls+df.total_eve_calls+
                            df.total_night_calls+df.total_intl_calls)/df.account_length
    return X, X2, y

def get_model(X,y):
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
