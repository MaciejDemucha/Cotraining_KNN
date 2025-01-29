import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel
 
scores = np.load("scores_n7_pinf.npy")

tablefmt="latex"
table = tabulate(np.mean(scores, axis=-1), 
                  tablefmt="latex_booktabs",
                  floatfmt=".4f",
                 headers=["KNN 3", "KNN 5", "cotrain knn","custom_cotrain knn", "selftrain svc"], 
                 showindex=["wisconsin", "iris"]
)

print(table)

table = tabulate(np.std(scores, axis=-1), 
                  tablefmt="latex_booktabs",
                  floatfmt=".4f",
                 headers=["KNN 3", "KNN 5", "cotrain knn","custom_cotrain knn", "selftrain svc"], 
                 showindex=["wisconsin", "iris"]
)

print(table)


# for each dataset
for i in range(scores.shape[0]):
    stat_mat = np.zeros((scores.shape[1], scores.shape[1]))
    for j in range(scores.shape[1]):
        for k in range(scores.shape[1]):
            t, p = ttest_rel(scores[i, j, :], scores[i, k, :])
            stat_mat[j, k] = p < 0.05
    
    table = tabulate(stat_mat)
    print(table)
