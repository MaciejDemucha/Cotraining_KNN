from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from classifiers import CoTrainingClassifier
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
		SelfTrainingClassifier(SVC(probability=True, gamma="auto"))]

	rng = np.random.RandomState(42)
	dataset = load_iris()
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

	# N_SAMPLES = 25000
	# N_FEATURES = 1000
	# X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)

	# y[:N_SAMPLES//2] = -1

	# X_test = X[-N_SAMPLES//4:]
	# y_test = y[-N_SAMPLES//4:]

	# X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
	# y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

	# y = y[:-N_SAMPLES//4]
	# X = X[:-N_SAMPLES//4]


	# X1 = X[:,:N_FEATURES // 2]
	# X2 = X[:, N_FEATURES // 2:]



	#print ('SVM')
	#base_svm = LinearSVC()
	#base_svm.fit(X_labeled, y_labeled)
	#y_pred = base_lr.predict(X_test)
	#print (classification_report(y_test, y_pred))
	
	#print ('SVM CoTraining')
	#svm_co_clf = CoTrainingClassifier(SVC(probability=True), u=N_SAMPLES//10)
	#svm_co_clf.fit(X1, X2, y)
	#y_pred = svm_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	#print (classification_report(y_test, y_pred))

	# print ('kNN')
	# base_kNN = KNeighborsClassifier(p=2)
	# base_kNN.fit(X_labeled, y_labeled)
	# y_pred = base_kNN.predict(X_test)
	# print (classification_report(y_test, y_pred))
	
	# print ('kNN CoTraining 2 2')
	# knn_co_clf = CoTrainingClassifier(clf=KNeighborsClassifier(p=2),clf2=KNeighborsClassifier(p=2), u=N_SAMPLES//10)
	# knn_co_clf.fit(X1, X2, y)
	# y_pred = knn_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	# print (classification_report(y_test, y_pred))

	# print ('kNN CoTraining 1 2')
	# knn_co_clf = CoTrainingClassifier(clf=KNeighborsClassifier(p=1),clf2=KNeighborsClassifier(p=2), u=N_SAMPLES//10)
	# knn_co_clf.fit(X1, X2, y)
	# y_pred = knn_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	# print (classification_report(y_test, y_pred))

	# print ('kNN CoTraining 2 1')
	# knn_co_clf = CoTrainingClassifier(clf=KNeighborsClassifier(p=2),clf2=KNeighborsClassifier(p=1), u=N_SAMPLES//10)
	# knn_co_clf.fit(X1, X2, y)
	# y_pred = knn_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	# print (classification_report(y_test, y_pred))
	


	svc = SVC(probability=True, gamma="auto")
	self_training_model = SelfTrainingClassifier(svc)
	self_training_model.fit(X, y)
	self_train_pred = self_training_model.predict(X)
	accuracy = accuracy_score(y, self_train_pred)
	print("Iris Accuracy:", accuracy)

	

	acc_vec = []
	acc_vec_log = []
	acc_vec_log_cot = []
	rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

	for train, test in rskf.split(X, y):
		clf = self_training_model
		clf.fit(X[train], y[train])
		y_pred = clf.predict(X[test])

		accuracy = accuracy_score(y[test], y_pred)
		acc_vec.append(accuracy)

		X_test = X[test]
		
		print ('Logistic')
		base_lr = LogisticRegression()
		base_lr.fit(X_labeled, y_labeled)
		y_pred = base_lr.predict(X[test])
		accuracy = accuracy_score(y[test], y_pred)
		acc_vec_log.append(accuracy)

		print ('Logistic CoTraining')
		lg_co_clf = CoTrainingClassifier(LogisticRegression())
		lg_co_clf.fit(X1, X2, y)
		y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
		accuracy = accuracy_score(y[test], y_pred)
		acc_vec_log_cot.append(accuracy)

	print("Accuracy mean self train svc:", np.mean(acc_vec))
	print("Accuracy std:", np.std(acc_vec))

	print("Accuracy mean logistic:", np.mean(acc_vec_log))
	print("Accuracy std:", np.std(acc_vec_log))

	print("Accuracy mean cotraing logistic:", np.mean(acc_vec_log_cot))
	print("Accuracy std:", np.std(acc_vec_log_cot))



	scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

	for dataset_idx, (X, y) in enumerate(DATASETS):
		for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
			for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
				clf = clone(clf_prot)
				X_test = X[test]
				X_train = X[train]

				X1_train = X_train[:,:N_FEATURES // 2]
				X2_train = X_train[:, N_FEATURES // 2:]
				X1_test = X_test[:,:N_FEATURES // 2]
				X2_test = X_test[:, N_FEATURES // 2:]
				#X1_train, X1_test = X1[train], X1[test]
				#X2_train, X2_test = X2[train], X2[test]
				if isinstance(clf, CoTrainingClassifier):
					clf.fit(X1_train, X2_train, y[train])
					y_pred = clf.predict(X1_test, X2_test)
				else:
					clf.fit(X[train], y[train])
					y_pred = clf.predict(X[test])
				score = accuracy_score(y[test], y_pred)
				scores[dataset_idx, classifier_idx, fold_idx] = score
			
	np.save("scores", scores)
	
