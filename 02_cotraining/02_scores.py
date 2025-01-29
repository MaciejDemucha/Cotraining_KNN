import numpy as np
from tabulate import tabulate
import os

from scipy.stats import ttest_rel


# for files .npy
for file in os.listdir():
    if file.endswith(".npy"):
        scores = np.load(file)
        print(file)
        table = tabulate(
            np.mean(scores, axis=-1),
            tablefmt="latex_booktabs",
            floatfmt=".4f",
            headers=["KNN 3", "KNN 5", "cotrain knn", "custom_cotrain knn", "selftrain svc"],
            showindex=["wisconsin", "iris"],
        )
        print('')
        print(table)

        table = tabulate(
            np.std(scores, axis=-1),
            tablefmt="latex_booktabs",
            floatfmt=".4f",
            headers=["KNN 3", "KNN 5", "cotrain knn", "custom_cotrain knn", "selftrain svc"],
            showindex=["wisconsin", "iris"],
        )

        print('\n')
        print(table)
        print('\n')


        # for each dataset
        for i in range(scores.shape[0]):
            stat_mat = np.zeros((scores.shape[1], scores.shape[1]))
            for j in range(scores.shape[1]):
                for k in range(scores.shape[1]):
                    t, p = ttest_rel(scores[i, j, :], scores[i, k, :])
                    stat_mat[j, k] = p < 0.05

            table = tabulate(stat_mat)
            # print(table)