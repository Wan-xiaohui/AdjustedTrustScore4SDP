from trustscore import trustscore_evaluation
from utils.helper import *
import pickle
import warnings
import sys
from trustscore import trustscore
import numpy as np
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")


RAND_SEED = 222
np.random.seed(RAND_SEED)

# 5*5交叉验证
n_splits = 5
n_repeats = 5


if __name__ == '__main__':
    folder_path = 'datasets/'
    data_list, label_list, fname = load_data(folder_path)
    files = os.listdir(folder_path)

    # 采用的分类器方法
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF']

    begin_num = 1
    end_num = 20

    for n in range(begin_num, end_num + 1):

        print("*" * 20)
        print('File: ' + fname[n - 1] + '...')
        print("*" * 20)

        data = data_list[n - 1]
        label = label_list[n - 1]

        tests_pkfile = open('dump/test_predictions/' + fname[n - 1] + '.pickle', 'rb')
        data_splits = pickle.load(tests_pkfile)
        test_predictions = pickle.load(tests_pkfile)
        test_confidences = pickle.load(tests_pkfile)

        aucs_lst = []
        for clf in clfs:
            extra_plot_title = fname[n-1] + " | " + clf + " | Identify Incorrect"
            percentile_levels = [0 + 0.5 * i for i in range(200)]

            signal_names = ["Trust Score", "Trust Score-D", "Trust Score-U"]
            signals = [trustscore.TrustScore(),
                       trustscore.TrustScore(k=5, alpha=0.1, filtering="density"),
                       trustscore.TrustScore(k=5, alpha=0.1, filtering="uncertainty")
                       ]
            trainer = trustscore_evaluation.run_clf

            all_auc, _, _, _, _ = trustscore_evaluation.run_precision_recall_experiment_RQ1(
                data,
                label,
                data_splits,
                test_predictions[clf],
                test_confidences[clf],
                percentile_levels=percentile_levels,
                signal_names=signal_names,
                signals=signals,
                extra_plot_title=extra_plot_title,
                predict_when_correct=False,
                legend_loc="upper left",
                skip_print=True
            )

            aucs_lst.append(all_auc)

        columns = ["Model Confidence"] + signal_names
        result_df = pd.DataFrame(aucs_lst, columns=columns)
        result_df.to_csv('dump/csvs/RQ1/Incorrect/'+fname[n - 1]+".csv")