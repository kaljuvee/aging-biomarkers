import operator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from eli5 import explain_prediction_df
from .plot import plot_importance2, plot_regression, plot_shap

def get_ct(df, gender=None, age=None):
    '''Grab a copy and filter the preprocessed data'''
    ct = df.copy()
    if gender is not None:
        ct = ct[ct['PATIENT_GNDR'] == gender]
    if age is not None:
        ct = ct[ct['PATIENT_AGE_YEARS'] > age]
    return ct

def predict_and_plot(ct, m):
    mask = ct[m].notnull()
    dataset = ct[mask]
    lim = (dataset[m].min(), dataset[m].max())
    Y = dataset[m].values
    X_ = dataset.drop(m, 1)
    X = X_.values
    # split data into train and test sets
    seed = 7
    #test_size = 0.33
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
        test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBRegressor()
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train)#, early_stopping_rounds=10, eval_metric="rmse", eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    mse = mean_squared_error(y_test, y_pred)
    plot_importance2(ct, model, m)
    plot_regression(y_pred, y_test)
    plot_shap(X_, model, m)
    return (X_test, y_test, y_pred,
            ClassifyModel(model, mse=mse))

def create_label(ct, marker, label, limit, op=operator.ge):
    ct[label] = op(ct[marker], limit)
    ct.drop(marker, 1, inplace=True)

def drop_all_but_in_included(ct, label, included_markers):
    ct.drop(set(ct.columns).difference(included_markers + [label,]),
            axis=1, inplace=True)
def predict_one(ct, marker, label, included_markers, regression=False):
    drop_all_but_in_included(ct, label, included_markers)
    return predict_and_plot(ct, label)

class ClassifyModel(object):
    def __init__(self, model, cm=None, precision=None, recall=None,
                 fbeta_score=None, support=None, mse=None):
        self.model = model
        self.cm = cm
        self.precision = precision
        self.recall = recall
        self.fbeta_score = fbeta_score
        self.support = support
        self.mse = mse

    def get_sens_spec(self):
        return (self.recall[0], self.recall[1])

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def predict_proba(self, data, **kwargs):
        return self.model.predict_proba(data, **kwargs)

class ModelConfig(object):
    def __init__(self, marker, label,
                 limit=None, op=None, gender=None,
                 low=None, high=None, default=True,
                 regression=False, age=None, absolute=False, excluded=None):
        self.marker = marker
        self.label = label
        self.limit = limit
        self.op = op
        self.gender = gender
        self.age = age
        self.low = low
        self.high = high
        self.model = None
        self.ct = None
        self.columns = None
        self.default = default
        self.regression = regression
        self.absolute = absolute
        self.excluded = set(excluded) if excluded is not None else set()
        self.pdp = {}

    def get_training(self, get_ct):
        ct = get_ct(gender=self.gender, age=self.age)
        ct = ct[np.isfinite(ct[self.marker])]
        ct.dropna(axis=1, how='all', inplace=True)
        if self.regression:
            ct[self.label] = ct[self.marker]
            ct.drop(self.marker, axis=1, inplace=True)
        elif self.low is None or self.high is None:
            create_label(ct, self.marker, self.label, self.limit, op=self.op)
        else:
            ct[self.label] = self.default
            for (markers, op) in zip((self.high, self.low), (operator.gt, operator.lt)):
                for (m, how) in markers.items():
                    limit = how if self.absolute else get_range(ct, m, how)
                    print('{} ({}), limit: {}'.format(m, how, limit))
                    ct[self.label] = self.op(ct[self.label], op(ct[m], limit))
        #for (n, c) in ct.items(): fix_missing(ct, c, n, {})
        self.ct = ct

    def fit(self, included_markers, get_ct):
        im = list(set(included_markers) - self.excluded)
        self.get_training(get_ct)
        ct = self.ct
        #print(self.ct[self.label].value_counts())
        (self.X_test, self.y_test,
         self.y_pred, self.model) = predict_one(ct, self.marker, self.label,
                                                im, regression=self.regression)
        #self.columns = [c for c in ct.columns if ct[c].notnull().any()][:-1]
        self.columns = ct.columns[:-1]

    def build_input(self, data):
        # XGBoost won't use a column if it only contains NaN
        columns = self.columns
        data_d = {}
        for c in columns:
            data_d[c] = [data.get(c, np.NaN)]
        return pd.DataFrame(data=data_d)[columns]

    def predict(self, data):
        X = self.build_input(data)
        result = self.model.predict(X.values)[0]
        if self.regression:
            return result
        return np.clip(result, 0, 1)

    def explain(self, data):
        data = self.build_input(data)
        X = data.iloc[0].values
        feature_names = [c for c in data.columns]
        return explain_prediction_df(self.model.model.get_booster(), X,
                                     feature_names=feature_names)
