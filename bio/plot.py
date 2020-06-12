from matplotlib import pyplot
import numpy as np
import seaborn as sns; sns.set(style='white', color_codes=True)
import scipy.stats as stats
import shap

def plot_importance2(ct, model, label, n=15):
    #columns = [c for c in ct.columns if ct[c].notnull().any()][:-1]
    columns = ct.columns[:-1]
    importance = model.get_booster().get_fscore()
    tuples = [(columns[int(k[1:])], importance[k]) for k in importance]
    tuples = sorted(tuples, key=lambda x: x[1], reverse=True)[:n]
    objects = [k for (k, v) in tuples]
    y_pos = np.arange(len(objects))
    values = [v for (k, v) in tuples]
    pyplot.figure()#figsize=(10, 15))
    pyplot.barh(y_pos, values, align='center')
    pyplot.yticks(y_pos, objects)
    pyplot.xlabel('score')
    title = f'{label} model feature importance'
    pyplot.title(title)
    pyplot.show()
    #pyplot.savefig(title)

def plot_regression(y_pred, y_test):
    lim = (y_pred.min(), y_pred.max())
    pyplot.figure()#figsize=(10, 15))
    g = sns.jointplot(y_pred, y_test, xlim=lim, ylim=lim, kind='reg')
    g.set_axis_labels('Predicted', 'Actual')
    g.annotate(stats.pearsonr)
    pyplot.show()

def plot_shap(X, model, label):
    shap_values = shap.TreeExplainer(model).shap_values(X.values)
    pyplot.figure()#figsize=(10, 15))
    title = f'SHAP summary {label}'
    pyplot.title(title)
    shap.summary_plot(shap_values, X)
    pyplot.show()
    title = f'SHAP dependence for {label}'
    for col in X.columns.difference(['PATIENT_AGE_YEARS', 'PATIENT_GNDR']):
        shap.dependence_plot(col, shap_values, X, interaction_index=None)
