import numpy as np
import random
import copy
from sklearn.base import ClassifierMixin, BaseEstimator

"""
	Parameters:
	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

class CustomCotrainingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator_1, estimator_2, p=-1, n=-1, k=30, u=75):
        super().__init__()
        self.estimator_1 = estimator_1
        self.estimator_2 = estimator_2
        self.p = p
        self.n = n
        self.k = k
        self.u = u

    def fit(self, X1, X2, y):
        labeled_mask = y != -1
        unlabeled_mask = ~labeled_mask
        X1_unlabeled, X2_unlabeled, y_unlabeled = X1[unlabeled_mask], X2[unlabeled_mask], y[unlabeled_mask]
        X1_labeled, X2_labeled, y_labeled = X1[labeled_mask], X2[labeled_mask], y[labeled_mask]

        U_prim = random.sample(y_unlabeled, self.u)

        y_unlabeled = [elem for elem in y_unlabeled if elem not in U_prim]

        L = [i for i, y_i in enumerate(y) if y_i != -1]
        
        for i in range(0, self.k):
            self.estimator_1.fit(X1_labeled, y_labeled)
            self.estimator_2.fit(X2_labeled, y_labeled)

            y1_prob = self.estimator_1.predict_proba(X1[U_prim])
            y2_prob = self.estimator_2.predict_proba(X2[U_prim])

            n1 = y1_prob[:,0].argsort()[:self.n]
            p1 = y1_prob[:,1].argsort()[-self.p:]

            n2 = y2_prob[:,0].argsort()[:self.n]
            p2 = y2_prob[:,1].argsort()[-self.p:]

            n = n1.extend(n2)
            p = p1.extend(p2)

            y[[U_prim[x] for x in p]] = 1
            y[[U_prim[x] for x in n]] = 0

            L.extend([U_prim[x] for x in p])
            L.extend([U_prim[x] for x in n])

            U_prim = [elem for elem in U_prim if not (elem in p or elem in n)]

            number_to_add = 2*(len(p) + len(n))
            for i in range(0, number_to_add):
                random_elem = random.choice(y_unlabeled)
                y_unlabeled.pop(y_unlabeled.index(random_elem))
                U_prim.append(random_elem)
                
        self.estimator_1.fit(X1[L], y[L])
        self.estimator_2.fit(X2[L], y[L])


    def predict(self, X1, X2):
        pass