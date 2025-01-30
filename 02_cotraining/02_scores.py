import numpy as np
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel


# for files .npy
for file in os.listdir():
    if file.endswith("scores_n3_p2_%03_t09.npy"):
        # if 'inf' not in file:
        #     p = int(file[8])
        #     print(p)
        #     n = int(file[11])
        #     print(n)
        #     label_percent = int(file[15])/10
        #     print(label_percent)
        #     threshold = int(file[19])/10
        #     print(threshold)
        # else:
        #     p, n, label_percent, threshold = 5, np.inf, 0.3, 0.9

        scores = np.load(file)
        print(file)
        table = tabulate(
            np.mean(scores, axis=-1),
            tablefmt="latex_booktabs",
            floatfmt=".4f",
            headers=["KNN 3", "KNN 5", "cotrain knn", "custom_cotrain knn", "selftrain svc"],
            showindex=["wisconsin", "iris", "diabetes"]
        )
        print('')
        print(table)

        table = tabulate(
            np.std(scores, axis=-1),
            tablefmt="latex_booktabs",
            floatfmt=".4f",
            headers=["KNN 3", "KNN 5", "cotrain knn", "custom_cotrain knn", "selftrain svc"],
            showindex=["wisconsin", "iris", "diabetes"]
        )

        print('\n')
        print(table)
        print('\n')


        dataset_names = ["Wisconsin", "Iris", "Diabetes"]
        model_names = ["KNN 3", "KNN 5", "Cotrain KNN", "Custom Cotrain KNN", "Selftrain SVC"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # for each dataset
        for i in range(scores.shape[0]):
            stat_mat = np.zeros((scores.shape[1], scores.shape[1]))
            for j in range(scores.shape[1]):
                for k in range(scores.shape[1]):
                    t, p = ttest_rel(scores[i, j, :], scores[i, k, :])
                    stat_mat[j, k] = p < 0.05
            
            sns.heatmap(stat_mat, annot=True, cmap="coolwarm", cbar=False, linewidths=0.5, ax=axes[i])
            axes[i].set_title(f"Macierz istotności - {dataset_names[i]}")
            axes[i].set_xticklabels(model_names, rotation=45, ha="right")
            axes[i].set_yticklabels(model_names, rotation=0)

            table = tabulate(stat_mat)
            print(table)

        plt.tight_layout()
        plt.savefig(f'./heatmap{file[:-4]}')
        # plt.show()
