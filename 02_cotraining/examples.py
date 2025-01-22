from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from classifiers import CoTrainingClassifier
from custom_cotraining import CustomCotrainingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold


if __name__ == '__main__':
	DATASETS = [
    load_breast_cancer(return_X_y=True),
    load_iris(return_X_y=True)]


	CLASSIFIERS = [
		KNeighborsClassifier(n_neighbors=3),
		KNeighborsClassifier(n_neighbors=15),
		CoTrainingClassifier(LogisticRegression()),
		CustomCotrainingClassifier(LogisticRegression(), LogisticRegression(), p=1, n=3, k=30, u=75),
		SelfTrainingClassifier(SVC(probability=True, gamma="auto"))]

	rng = np.random.RandomState(42)
	dataset = load_breast_cancer()
	random_unlabeled_points = rng.rand(dataset.target.shape[0]) < 0.3
	dataset.target[random_unlabeled_points] = -1
	X, y = dataset.data, dataset.target
	feature_names = dataset.feature_names
	target_names = dataset.target_names

	mask_labeled = y != -1  # Mask for labeled points
	X_labeled = X[mask_labeled]
	y_labeled = y[mask_labeled]

	N_SAMPLES, N_FEATURES = X.shape

	X1 = X[:,:N_FEATURES // 2]
	X2 = X[:, N_FEATURES // 2:]
	

	rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

	scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

	for dataset_idx, (X, y) in enumerate(DATASETS):
		rng = np.random.RandomState(42)

		feature_names = dataset.feature_names if hasattr(dataset, "feature_names") else None
		target_names = dataset.target_names if hasattr(dataset, "target_names") else None

		# Create random unlabeled points
		random_unlabeled_points = rng.rand(y.shape[0]) < 0.3
		y[random_unlabeled_points] = -1

		mask_labeled = y != -1

		N_SAMPLES, N_FEATURES = X.shape

		for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
			for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
				clf = clone(clf_prot)
				X_test = X[test]
				X_train = X[train]

				X1_train = X_train[:,:N_FEATURES // 2]
				X2_train = X_train[:, N_FEATURES // 2:]
				X1_test = X_test[:,:N_FEATURES // 2]
				X2_test = X_test[:, N_FEATURES // 2:]
				if isinstance(clf, CoTrainingClassifier) | isinstance(clf, CustomCotrainingClassifier):
					clf.fit(X1_train, X2_train, y[train])
					y_pred = clf.predict(X1_test, X2_test)
				else:
					clf.fit(X[train], y[train])
					y_pred = clf.predict(X[test])
				score = accuracy_score(y[test], y_pred)
				scores[dataset_idx, classifier_idx, fold_idx] = score
			
	np.save("scores", scores)
	
