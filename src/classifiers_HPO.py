import numpy as np
import sys
sys.dont_write_bytecode = True
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, space_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef


N_SPLITS = 5
MAX_EVALS = 20


def calc_performance(label_true, label_pred):
    MCC = matthews_corrcoef(label_true, label_pred)
    return MCC


def KNN_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):

    def KNN_hyperopt_train_val(params):
        clf = neighbors.KNeighborsClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)
            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'n_neighbors': hp.randint('n_neighbors', 1, 11),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'p': hp.randint('p', 1, 6),
        'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree'])
    }

    if opt_algo == 'RAND':
        best = fmin(KNN_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(KNN_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    else:
        print('PLEASE SET YOUR OPTIMIZATION ALGORITHM !!!')

    params = space_eval(param_space, best)
    model_tune = neighbors.KNeighborsClassifier(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def LR_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def LR_hyperopt_train_val(params):
        clf = LogisticRegression(**params)
        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'C': hp.loguniform('C', -10, 10),
        'penalty': 'elasticnet',
        'solver': 'saga',
        'l1_ratio': hp.uniform('alpha', 0, 1),
    }

    if opt_algo == 'RAND':
        best = fmin(LR_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(LR_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)
    model_tune = LogisticRegression(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune

    # pred_proba = model_tune.predict_proba(test_data)[:, 1]
    # pred_y = model_tune.predict(test_data)
    #
    # prec, recall, false_alarm, auc_value, F_measure, G_measure, bal = calc_performance(test_label, pred_y, pred_proba)
    #
    # if metrics == 'F_measure':
    #     result = F_measure
    #
    # elif metrics == 'AUC_PR':
    #     result = auc_value
    #
    # elif metrics == 'G_measure':
    #     result = G_measure
    #
    # elif metrics == 'Bal_value':
    #     result = bal
    #
    # else:
    #     print('PLEASE SELECT THE METRICS !!!')
    #
    # return model_tune, result


def SVM_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def SVM_hyperopt_train_val(params):
        clf = SVC(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),
        'C': hp.lognormal('C', -1, 11),
        'gamma': hp.choice('gamma', [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 'auto']),
        'probability': True,
        'max_iter': 5000,
    }

    if opt_algo == 'RAND':
        best = fmin(SVM_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(SVM_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)
    model_tune = SVC(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def MLP_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def MLP_hyperopt_train_val(params):
        clf = MLPClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'alpha': hp.choice('alpha', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
        'max_iter': 50,
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(i,) for i in range(5, 12, 2)] +
                                        [(i, j) for i in range(5, 12, 2) for j in range(5, 12, 2)])
    }

    if opt_algo == 'RAND':
        best = fmin(MLP_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(MLP_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)

    model_tune = MLPClassifier(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def DT_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def DT_hyperopt_train_val(params):
        clf = DecisionTreeClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {'criterion': hp.choice('criterion', ['gini', 'entropy']),
                   'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                   'min_samples_split': hp.randint('min_samples_split', 2, 6),
                   'min_samples_leaf': hp.randint('min_samples_leaf', 1, 6),
                   }

    if opt_algo == 'RAND':
        best = fmin(DT_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(DT_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)
    model_tune = DecisionTreeClassifier(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune

    # pred_proba = model_tune.predict_proba(test_data)[:, 1]
    # pred_y = model_tune.predict(test_data)
    #
    # prec, recall, false_alarm, auc_value, F_measure, G_measure, bal = calc_performance(test_label, pred_y, pred_proba)
    #
    # if metrics == 'F_measure':
    #     result = F_measure
    #
    # elif metrics == 'AUC_PR':
    #     result = auc_value
    #
    # elif metrics == 'G_measure':
    #     result = G_measure
    #
    # elif metrics == 'Bal_value':
    #     result = bal
    #
    # else:
    #     print('PLEASE SELECT THE METRICS !!!')
    #
    # return model_tune, result


def RF_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def RF_hyperopt_train_val(params):
        clf = RandomForestClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)
            pred_y = clf.predict(val_X)
            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'n_estimators': hp.choice('n_estimators', [10 * i for i in range(1, 6)]),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.7]),
        'min_samples_split': hp.randint('min_samples_split', 2, 6),
        'warm_start': True
    }

    if opt_algo == 'RAND':
        best = fmin(RF_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(RF_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)
    model_tune = RandomForestClassifier(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def GB_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def GB_hyperopt_train_val(params):
        clf = GradientBoostingClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)
            pred_y = clf.predict(val_X)
            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {'n_estimators': hp.choice('n_estimators', [10 * i for i in range(1, 6)]),
                   'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.7]),
                   'min_samples_split': hp.randint('min_samples_split', 2, 6),
                   'learning_rate': hp.choice('learning_rate', [0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
                   }

    if opt_algo == 'RAND':
        best = fmin(GB_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(GB_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)
    model_tune = GradientBoostingClassifier(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def NB_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def NB_hyperopt_train_val(params):
        clf = GaussianNB(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'var_smoothing': hp.loguniform('var_smoothing', -10, -1)
    }

    if opt_algo == 'RAND':
        best = fmin(NB_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS)

    elif opt_algo == 'TPE':
        best = fmin(NB_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS)

    params = space_eval(param_space, best)
    model_tune = GaussianNB(**params)

    train_data, train_label = SMOTE(sampling_strategy='auto', k_neighbors=3).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune