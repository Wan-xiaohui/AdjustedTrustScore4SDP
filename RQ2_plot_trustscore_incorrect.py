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


def load_jit_data(folder_path):
    # Initialize lists to store features and labels
    data_list = []
    label_list = []
    fname_list = []

    # Iterate over each CSV file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file using pandas
            df = pd.read_csv(file_path, index_col=None, header=0,  sep='[:,;]', engine='python')
            # pd.read_csv(file_path)
            df.dropna(inplace=True)

            # Extract features (excluding first two columns and last column)
            features = df.values[:, :-1]

            # Extract labels (last column)
            labels = df.values[:, -1].astype('int')

            # Append features and labels to the lists
            data_list.append(features)
            label_list.append(labels)
            fname_list.append(filename[:-4])

    return data_list, label_list, fname_list


if __name__ == '__main__':
    folder_path = 'datasets/'
    data_list, label_list, fname = load_data(folder_path)

    jit_folder_path = 'datasets/NEW-JIT/'
    jit_data_list, jit_label_list, jit_fname = load_jit_data(jit_folder_path)

    data_list.extend(jit_data_list)
    label_list.extend(jit_label_list)
    fname.extend(jit_fname)

    # 采用的分类器方法
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']

    for n in range(len(fname)):

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
            extra_plot_title = fname[n] + " | " + clf + " | Identify Incorrect"
            percentile_levels = [0 + 0.5 * i for i in range(200)]

            signal_names = ["Adjusted Trust Score", "Trust Score"]
            signals = [trustscore.TrustScore(),
                       trustscore.TrustScore()
                       ]
            all_auc, _, _, _, _ = trustscore_evaluation.run_precision_recall_experiment_RQ2(
                data,
                label,
                data_splits,
                test_predictions[clf],
                test_confidences[clf],
                percentile_levels=percentile_levels,
                signal_names=signal_names,
                signals=signals,
                extra_plot_title=extra_plot_title,
                skip_print=True,
                predict_when_correct=False,
                fig_path="dump/plots/RQ2/Incorrect/"
            )

            aucs_lst.append(all_auc)

        columns = ["Model Confidence"] + signal_names
        result_df = pd.DataFrame(aucs_lst, columns=columns)
        result_df.to_csv('dump/csvs/RQ2/Incorrect/'+fname[n]+".csv")