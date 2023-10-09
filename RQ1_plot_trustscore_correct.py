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


if __name__ == '__main__':
    folder_path = 'datasets/'
    data_list, label_list, fname = load_data(folder_path)
    files = os.listdir(folder_path)

    # 采用的分类器方法
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF']

    for n in range(20):

        print("*" * 20)
        print('File: ' + fname[n] + '...')
        print("*" * 20)

        data = data_list[n]
        label = label_list[n]

        tests_pkfile = open('dump/test_predictions/' + fname[n] + '.pickle', 'rb')
        data_splits = pickle.load(tests_pkfile)
        test_predictions = pickle.load(tests_pkfile)
        test_confidences = pickle.load(tests_pkfile)

        aucs_lst = []
        for clf in clfs:
            extra_plot_title = fname[n] + " | " + clf + " | Identify Correct"
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
                predict_when_correct=True,
                legend_loc="upper left",
                skip_print=True
            )

            aucs_lst.append(all_auc)

        columns = ["Model Confidence"] + signal_names
        result_df = pd.DataFrame(aucs_lst, columns=columns)
        result_df.to_csv('dump/csvs/RQ1/Correct/'+fname[n - 1]+".csv")
