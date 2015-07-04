from sklearn.cross_validation import ShuffleSplit
import pandas as pd
import numpy as np
from feature_extractor import FeatureExtractor
from regressor import Regressor

def train_model(X_df, y_array, skf_is):
    fe = FeatureExtractor()
    fe.fit(X_df, y_array)
    X_array = fe.transform(X_df)
    # Regression
    train_is, _ = skf_is
    X_train_array = np.array([X_array[i] for i in train_is])
    y_train_array = np.array([y_array[i] for i in train_is])
    reg = Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg

def test_model(trained_model, X_df, skf_is):
    fe, reg = trained_model
    # Feature extraction
    X_array = fe.transform(X_df)
    # Regression
    _, test_is = skf_is
    X_test_array = np.array([X_array[i] for i in test_is])
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array

data = pd.read_csv("data_amadeus.csv")
X_df = data.drop(['PAX', 'log_PAX'], axis=1)
y_array = data['log_PAX'].values

skf = ShuffleSplit( y_array.shape[0], n_iter=2, test_size=0.5, random_state=57)
skf_is = list(skf)[0]

trained_model = train_model(X_df, y_array, skf_is)
y_pred_array = test_model(trained_model, X_df, skf_is)
_, test_is = skf_is
ground_truth_array = y_array[test_is]

score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
print score
