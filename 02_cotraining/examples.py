import pandas as pd
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
from scipy.io import arff

# to trzeba zastosować w tych z arff
# for dataset_idx, file in enumerate(CUSTOM_DATASETS_FILES):
# 		rng = np.random.RandomState(42)

# 		data, meta = arff.loadarff(file)
# 		df = pd.DataFrame(data)
# 		# If any columns are of type 'object' (byte strings), convert them to strings
# 		for col in df.select_dtypes([object]).columns:
# 			df[col] = df[col].str.decode('utf-8')  # Convert bytes to string
# 			df[col] = pd.to_numeric(df[col], errors='ignore')  # Convert to int/float if possible

# 		X = df.iloc[:, :-1].to_numpy()
# 		y = df.iloc[:, -1].to_numpy()


def arff_to_sklearn(arff_file):
    """
    Converts an ARFF file to a Pandas DataFrame for use with sklearn.

    Parameters:
        arff_file (str): Path to the .arff file.

    Returns:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target array (if available), otherwise None.
        feature_names (list): Names of the features.
        target_name (str or None): Name of the target column (if available).
    """
    data, meta = arff.loadarff(arff_file)
    df = pd.DataFrame(data)

    # Convert byte strings to regular strings for categorical attributes
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Identify the target column (assuming the last column is the target)
    feature_names = list(df.columns[:-1])
    target_name = df.columns[-1] if df.shape[1] > 1 else None

    # Extract features and target
    X = df.iloc[:, :-1].values if df.shape[1] > 1 else df.values
    y = df.iloc[:, -1].values if df.shape[1] > 1 else None

    return X, y


# Example usage:
# X, y, feature_names, target_name = arff_to_sklearn('your_file.arff')


def ucz_sie_maszynowo_i_zapisz_wyniki(
    DATASETS, DATA_LABEL_PERCENT, THRESHOLD, neighbors, p_metric, CLASSIFIERS
):
    rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

    scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

    for dataset_idx, (X, y) in enumerate(DATASETS):
        print(type(X[0][0]), type(y[0]))
        rng = np.random.RandomState(42)

        # Create random unlabeled points
        random_unlabeled_points = rng.rand(y.shape[0]) < DATA_LABEL_PERCENT
        y[random_unlabeled_points] = -1

        mask_labeled = y != -1

        N_SAMPLES, N_FEATURES = X.shape
        # for dataset_idx, file in enumerate(DATASETS):
        #     rng = np.random.RandomState(42)

        #     data, meta = arff.loadarff(file)
        #     df = pd.DataFrame(data)
        #     # If any columns are of type 'object' (byte strings), convert them to strings
        #     for col in df.select_dtypes([object]).columns:
        #         df[col] = df[col].str.decode('utf-8')  # Convert bytes to string
        #         df[col] = pd.to_numeric(df[col], errors='ignore')  # Convert to int/float if possible

        #     X = df.iloc[:, :-1].to_numpy()
        #     y = df.iloc[:, -1].to_numpy()
        #     random_unlabeled_points = rng.rand(y.shape[0]) < DATA_LABEL_PERCENT
        #     y[random_unlabeled_points] = -1

        #     mask_labeled = y != -1

        #     N_SAMPLES, N_FEATURES = X.shape

        for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
            for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
                clf = clone(clf_prot)
                X_test = X[test]
                X_train = X[train]

                X1_train = X_train[:, : N_FEATURES // 2]
                X2_train = X_train[:, N_FEATURES // 2 :]
                X1_test = X_test[:, : N_FEATURES // 2]
                X2_test = X_test[:, N_FEATURES // 2 :]
                if isinstance(clf, CustomCotrainingClassifier):
                    clf.fit(X1_train, X2_train, y[train], threshold=THRESHOLD)
                    y_pred = clf.predict(X1_test, X2_test)
                elif isinstance(clf, CoTrainingClassifier):
                    clf.fit(X1_train, X2_train, y[train])
                    y_pred = clf.predict(X1_test, X2_test)
                else:
                    clf.fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                score = accuracy_score(y[test], y_pred)
                scores[dataset_idx, classifier_idx, fold_idx] = score

    np.save(
        f"scores_n{neighbors}_p{p_metric}_%{str(DATA_LABEL_PERCENT).replace('.', '')}_t{str(THRESHOLD).replace('.', '')}",
        scores,
    )


if __name__ == "__main__":
    DATASETS = [
        load_breast_cancer(return_X_y=True),
        load_iris(return_X_y=True),
        arff_to_sklearn("diabetes.arff"),
    ]

    CUSTOM_DATASETS_FILES = [
        "diabetes.arff",
        # "blood.arff"
    ]

    DATA_LABEL_PERCENT = 0.3
    THRESHOLD = 0.9

    NEIGHBORS = 5
    P_METRIC = 2
    knn1 = KNeighborsClassifier(n_neighbors=NEIGHBORS, p=P_METRIC)
    knn2 = KNeighborsClassifier()

    CLASSIFIERS = [
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        CoTrainingClassifier(knn1, knn2),
        CustomCotrainingClassifier(knn1, knn2, p=1, n=3, k=30, u=75),
        SelfTrainingClassifier(SVC(probability=True, gamma="auto")),
    ]

    rng = np.random.RandomState(42)
    dataset = load_breast_cancer()
    random_unlabeled_points = rng.rand(dataset.target.shape[0]) < DATA_LABEL_PERCENT
    dataset.target[random_unlabeled_points] = -1
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names
    target_names = dataset.target_names

    mask_labeled = y != -1  # Mask for labeled points
    X_labeled = X[mask_labeled]
    y_labeled = y[mask_labeled]

    N_SAMPLES, N_FEATURES = X.shape

    X1 = X[:, : N_FEATURES // 2]
    X2 = X[:, N_FEATURES // 2 :]

    for label_percent in [0.1, 0.2, 0.3]:
        ucz_sie_maszynowo_i_zapisz_wyniki(
            DATASETS, label_percent, THRESHOLD, NEIGHBORS, P_METRIC, CLASSIFIERS
        )

    for threshold in [0.7, 0.8, 0.9]:
        ucz_sie_maszynowo_i_zapisz_wyniki(
            DATASETS, DATA_LABEL_PERCENT, threshold, NEIGHBORS, P_METRIC, CLASSIFIERS
        )

    for p_value in [1, 2, np.inf]:
        ucz_sie_maszynowo_i_zapisz_wyniki(
            DATASETS, DATA_LABEL_PERCENT, THRESHOLD, NEIGHBORS, p_value, CLASSIFIERS
        )

    for n_value in [3, 5, 7]:
        ucz_sie_maszynowo_i_zapisz_wyniki(
            DATASETS, DATA_LABEL_PERCENT, THRESHOLD, n_value, P_METRIC, CLASSIFIERS
        )