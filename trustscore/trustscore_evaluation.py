# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import auc


class expFunc(object):
    def __init__(self, alpha=1.0, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, x):
        return np.exp(self.alpha * (x - self.beta))


def run_clf(model, X_test):
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    return y_pred, confidences


def plot_precision_curve(extra_plot_title,
                         percentile_levels,
                         signal_names,
                         final_TPs,
                         final_stderrs,
                         final_misclassification,
                         colors=["blue", "darkorange", "brown", "red", "purple"],
                         legend_loc=None,
                         figure_size=None,
                         ylim=None,
                         fig_path="dump/plots/RQ1/Correct/"
                         ):

    if figure_size is not None:
        plt.figure(figsize=figure_size)

    title = "Precision Curve" if extra_plot_title == "" else extra_plot_title
    plt.title(title, fontsize=20)
    colors = colors + list(cm.rainbow(np.linspace(0, 1, len(final_TPs))))

    plt.xlabel("Percentile level", fontsize=18)
    plt.ylabel("Precision", fontsize=18)

    auc_lst = []
    for i, signal_name in enumerate(signal_names):
        ls = "--" if ("Model" in signal_name) else "-"
        plt.plot(percentile_levels, final_TPs[i], ls, c=colors[i], label=signal_name)

        if final_stderrs is not None:
            plt.fill_between(percentile_levels,
                             final_TPs[i] - final_stderrs[i],
                             final_TPs[i] + final_stderrs[i],
                             color=colors[i],
                             alpha=0.1
                             )
        auc_value = auc(np.array(percentile_levels)/100, final_TPs[i])
        auc_lst.append(auc_value)

    if legend_loc is None:
        if 0. in percentile_levels:
            plt.legend(loc="lower right", fontsize=14)
        else:
            plt.legend(loc="upper left", fontsize=14)

    else:
        if legend_loc == "outside":
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=14)
        else:
            plt.legend(loc=legend_loc, fontsize=14)

    if ylim is not None:
        plt.ylim(*ylim)

    model_acc = 100 * (1 - final_misclassification)
    plt.axvline(x=model_acc, linestyle="dotted", color="black")

    fig_name = extra_plot_title.replace(' | ', '_')
    plt.savefig(fig_path + fig_name + '.pdf')
    plt.cla()

    return auc_lst


def run_precision_recall_experiment_RQ1(X,
                                        y,
                                        data_splits,
                                        test_predictions,
                                        test_confidences,
                                        percentile_levels,
                                        extra_plot_title="",
                                        signals=[],
                                        signal_names=[],
                                        predict_when_correct=False,
                                        legend_loc="upper left",
                                        skip_print=True,
                                        fig_path="dump/plots/RQ1/Correct/"
                                        ):

    def get_stderr(L):
        return np.std(L) / np.sqrt(len(L))

    all_signal_names = ["Model Confidence"] + signal_names
    all_TPs = [[[] for p in percentile_levels] for signal in all_signal_names]
    misclassifications = []
    sign = 1 if predict_when_correct else -1

    n_repeats = len(data_splits)
    n_splits = len(data_splits[0]['train_inds'])

    for n_repeat in range(n_repeats):
        for n_split in range(n_splits):

            train_ind = data_splits[n_repeat]['train_inds'][n_split]
            test_ind = data_splits[n_repeat]['test_inds'][n_split]

            X_train, y_train = X[train_ind, :], y[train_ind]
            X_test, y_test = X[test_ind, :], y[test_ind]

            testing_prediction = (test_predictions[test_ind, n_repeat] > 0.5).astype(np.int)

            target_points = np.where(testing_prediction == y_test)[0] if predict_when_correct else np.where(testing_prediction != y_test)[0]

            testing_confidence_raw = test_confidences[test_ind, n_repeat]
            final_signals = [testing_confidence_raw]

            for signal in signals:
                signal.fit(X_train, y_train)
                final_signals.append(signal.get_score(X_test, testing_prediction))

            for p, percentile_level in enumerate(percentile_levels):
                all_high_confidence_points = [np.where(sign*signal >= np.percentile(sign*signal, percentile_level))[0]
                                              for signal in final_signals]

                if 0 in map(len, all_high_confidence_points):
                    continue

                TP = [len(np.intersect1d(high_confidence_points, target_points)) / (1. * len(high_confidence_points))
                      for high_confidence_points in all_high_confidence_points]

                for i in range(len(all_signal_names)):
                    all_TPs[i][p].append(TP[i])

            misclassifications.append(len(target_points) / (1. * len(X_test)))

    final_TPs = [[] for signal in all_signal_names]
    final_stderrs = [[] for signal in all_signal_names]

    for p, percentile_level in enumerate(percentile_levels):
        for i in range(len(all_signal_names)):
            final_TPs[i].append(np.mean(all_TPs[i][p]))
            final_stderrs[i].append(get_stderr(all_TPs[i][p]))

        if not skip_print:
            print("Precision at percentile", percentile_level)
            ss = ""
            for i, signal_name in enumerate(all_signal_names):
                ss += (signal_name + (": %.4f  " % final_TPs[i][p]))
            print(ss)
            print()

    final_misclassification = np.mean(misclassifications)

    if not skip_print:
        print("Misclassification rate mean/std", np.mean(misclassifications), get_stderr(misclassifications))

    for i in range(len(all_signal_names)):
        final_TPs[i] = np.array(final_TPs[i])
        final_stderrs[i] = np.array(final_stderrs[i])

    all_auc = plot_precision_curve(extra_plot_title,
                                   percentile_levels,
                                   all_signal_names,
                                   final_TPs,
                                   final_stderrs,
                                   final_misclassification,
                                   ["blue", "darkorange", "brown", "red", "purple"],
                                   legend_loc,
                                   fig_path=fig_path
                                   )

    return (all_auc, all_signal_names, final_TPs, final_stderrs, final_misclassification)


def run_precision_recall_experiment_RQ1_cross(X_train,
                                              y_train,
                                              X_test,
                                              y_test,
                                              test_predictions,
                                              test_confidences,
                                              percentile_levels,
                                              extra_plot_title="",
                                              signals=[],
                                              signal_names=[],
                                              predict_when_correct=False,
                                              legend_loc="upper left",
                                              skip_print=True,
                                              fig_path="dump/plots/RQ1_cross/Correct/"
                                              ):

    all_signal_names = ["Model Confidence"] + signal_names
    all_TPs = [[[] for p in percentile_levels] for signal in all_signal_names]
    misclassifications = []
    sign = 1 if predict_when_correct else -1

    testing_prediction = (test_predictions > 0.5).astype(np.int)
    target_points = np.where(testing_prediction == y_test)[0] \
        if predict_when_correct else np.where(testing_prediction != y_test)[0]

    testing_confidence_raw = test_confidences
    final_signals = [testing_confidence_raw]

    for signal in signals:
        signal.fit(X_train, y_train)
        final_signals.append(signal.get_score(X_test, testing_prediction))

    for p, percentile_level in enumerate(percentile_levels):
        all_high_confidence_points = [np.where(sign * signal >= np.percentile(sign * signal, percentile_level))[0]
                                      for signal in final_signals]

        if 0 in map(len, all_high_confidence_points):
            continue

        TP = [len(np.intersect1d(high_confidence_points, target_points)) / (1. * len(high_confidence_points))
              for high_confidence_points in all_high_confidence_points]

        for i in range(len(all_signal_names)):
            all_TPs[i][p].append(TP[i])

    misclassifications.append(len(target_points) / (1. * len(X_test)))

    final_TPs = [[] for signal in all_signal_names]

    for p, percentile_level in enumerate(percentile_levels):
        for i in range(len(all_signal_names)):
            final_TPs[i].append(np.mean(all_TPs[i][p]))

        if not skip_print:
            print("Precision at percentile", percentile_level)
            ss = ""
            for i, signal_name in enumerate(all_signal_names):
                ss += (signal_name + (": %.4f  " % final_TPs[i][p]))
            print(ss)
            print()

    final_misclassification = np.mean(misclassifications)

    for i in range(len(all_signal_names)):
        final_TPs[i] = np.array(final_TPs[i])

    nan_indices = np.where(np.isnan(final_TPs[0]))[0]
    final_TPs = [arr[~np.isnan(arr)] for arr in final_TPs]

    percentile_levels = [x for i, x in enumerate(percentile_levels) if i not in nan_indices]

    all_auc = plot_precision_curve(extra_plot_title,
                                   percentile_levels,
                                   all_signal_names,
                                   final_TPs,
                                   None,
                                   final_misclassification,
                                   ["blue", "darkorange", "brown", "red", "purple"],
                                   legend_loc,
                                   fig_path=fig_path
                                   )

    return (all_auc, all_signal_names, final_TPs, final_misclassification)


def run_precision_recall_experiment_RQ2(X,
                                        y,
                                        data_splits,
                                        test_predictions,
                                        test_confidences,
                                        percentile_levels,
                                        extra_plot_title="",
                                        signals=[],
                                        signal_names=[],
                                        predict_when_correct=False,
                                        legend_loc="upper left",
                                        skip_print=True,
                                        fig_path="dump/plots/RQ2/Correct/"
                                        ):

    def get_stderr(L):
        return np.std(L) / np.sqrt(len(L))

    all_signal_names = ["Model Confidence"] + signal_names
    all_TPs = [[[] for p in percentile_levels] for signal in all_signal_names]
    misclassifications = []
    sign = 1 if predict_when_correct else -1

    n_repeats = len(data_splits)
    n_splits = len(data_splits[0]['train_inds'])

    for n_repeat in range(n_repeats):
        for n_split in range(n_splits):

            train_ind = data_splits[n_repeat]['train_inds'][n_split]
            test_ind = data_splits[n_repeat]['test_inds'][n_split]

            X_train, y_train = X[train_ind, :], y[train_ind]
            X_test, y_test = X[test_ind, :], y[test_ind]

            testing_prediction = (test_predictions[test_ind, n_repeat] > 0.5).astype(np.int)
            target_points = np.where(testing_prediction == y_test)[0] if predict_when_correct else np.where(testing_prediction != y_test)[0]

            testing_confidence_raw = test_confidences[test_ind, n_repeat]
            final_signals = [testing_confidence_raw]

            for indx, signal in enumerate(signals):

               signal.fit(X_train, y_train)

               if indx == 0:
                   exp_func = expFunc()
                   final_signals.append(exp_func.evaluate(testing_confidence_raw) * signal.get_score(X_test, testing_prediction))

               else:
                   final_signals.append(signal.get_score(X_test, testing_prediction))

            for p, percentile_level in enumerate(percentile_levels):
                all_high_confidence_points = [np.where(sign * signal >= np.percentile(sign * signal, percentile_level))[0]
                                              for signal in final_signals]

                if 0 in map(len, all_high_confidence_points):
                    continue

                TP = [len(np.intersect1d(high_confidence_points, target_points)) / (1. * len(high_confidence_points))
                      for high_confidence_points in all_high_confidence_points]

                for i in range(len(all_signal_names)):
                    all_TPs[i][p].append(TP[i])

            misclassifications.append(len(target_points) / (1. * len(X_test)))

    final_TPs = [[] for signal in all_signal_names]
    final_stderrs = [[] for signal in all_signal_names]

    for p, percentile_level in enumerate(percentile_levels):
        for i in range(len(all_signal_names)):
            final_TPs[i].append(np.mean(all_TPs[i][p]))
            final_stderrs[i].append(get_stderr(all_TPs[i][p]))

        if not skip_print:
            print("Precision at percentile", percentile_level)
            ss = ""
            for i, signal_name in enumerate(all_signal_names):
                ss += (signal_name + (": %.4f  " % final_TPs[i][p]))
            print(ss)
            print()

    final_misclassification = np.mean(misclassifications)

    if not skip_print:
        print("Misclassification rate mean/std", np.mean(misclassifications), get_stderr(misclassifications))

    for i in range(len(all_signal_names)):
        final_TPs[i] = np.array(final_TPs[i])
        final_stderrs[i] = np.array(final_stderrs[i])

    all_auc = plot_precision_curve(extra_plot_title,
                                   percentile_levels,
                                   all_signal_names,
                                   final_TPs,
                                   final_stderrs,
                                   final_misclassification,
                                   ["blue", "darkorange", "brown", "red", "purple"],
                                   legend_loc,
                                   fig_path=fig_path
                                   )

    return (all_auc, all_signal_names, final_TPs, final_stderrs, final_misclassification)


def run_precision_recall_experiment_RQ2_cross(X_train,
                                              y_train,
                                              X_test,
                                              y_test,
                                              test_predictions,
                                              test_confidences,
                                              percentile_levels,
                                              extra_plot_title="",
                                              signals=[],
                                              signal_names=[],
                                              predict_when_correct=False,
                                              legend_loc="upper left",
                                              skip_print=True,
                                              fig_path="dump/plots/RQ2_cross/Correct/"
                                              ):

    all_signal_names = ["Model Confidence"] + signal_names
    all_TPs = [[[] for p in percentile_levels] for signal in all_signal_names]
    misclassifications = []
    sign = 1 if predict_when_correct else -1

    testing_prediction = (test_predictions > 0.5).astype(np.int)
    target_points = np.where(testing_prediction == y_test)[0] if predict_when_correct else np.where(testing_prediction != y_test)[0]

    testing_confidence_raw = test_confidences
    final_signals = [testing_confidence_raw]

    for indx, signal in enumerate(signals):

       signal.fit(X_train, y_train)

       if indx == 0:
           exp_func = expFunc()
           final_signals.append(exp_func.evaluate(testing_confidence_raw) * signal.get_score(X_test, testing_prediction))
           # final_signals.append(1.0 * signal.get_score(X_test, testing_prediction))

       else:
           final_signals.append(signal.get_score(X_test, testing_prediction))

    for p, percentile_level in enumerate(percentile_levels):
        all_high_confidence_points = [np.where(sign * signal >= np.percentile(sign * signal, percentile_level))[0] for signal in final_signals]

        if 0 in map(len, all_high_confidence_points):
            continue

        TP = [len(np.intersect1d(high_confidence_points, target_points)) / (1. * len(high_confidence_points))
              for high_confidence_points in all_high_confidence_points]

        for i in range(len(all_signal_names)):
            all_TPs[i][p].append(TP[i])

    misclassifications.append(len(target_points) / (1. * len(X_test)))

    final_TPs = [[] for signal in all_signal_names]

    for p, percentile_level in enumerate(percentile_levels):
        for i in range(len(all_signal_names)):
            final_TPs[i].append(np.mean(all_TPs[i][p]))

        if not skip_print:
            print("Precision at percentile", percentile_level)
            ss = ""
            for i, signal_name in enumerate(all_signal_names):
                ss += (signal_name + (": %.4f  " % final_TPs[i][p]))
            print(ss)
            print()

    final_misclassification = np.mean(misclassifications)

    for i in range(len(all_signal_names)):
        final_TPs[i] = np.array(final_TPs[i])

    nan_indices = np.where(np.isnan(final_TPs[0]))[0]
    final_TPs = [arr[~np.isnan(arr)] for arr in final_TPs]

    # 剔除 percentile_levels 对应索引位置的元素
    percentile_levels = [x for i, x in enumerate(percentile_levels) if i not in nan_indices]

    all_auc = plot_precision_curve(extra_plot_title,
                                   percentile_levels,
                                   all_signal_names,
                                   final_TPs,
                                   None,
                                   final_misclassification,
                                   ["blue", "darkorange", "brown", "red", "purple"],
                                   legend_loc,
                                   fig_path=fig_path
                                   )

    return (all_auc, all_signal_names, final_TPs, final_misclassification)