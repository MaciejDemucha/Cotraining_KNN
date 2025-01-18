from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y, check_array
import numpy as np

class CoTrainingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator_1, base_estimator_2, max_iter=10, confidence_threshold=0.75):
        """
        Initialize the Co-Training classifier.

        Parameters:
        - base_estimator_1: The first base classifier (e.g., LogisticRegression).
        - base_estimator_2: The second base classifier.
        - max_iter: Maximum number of co-training iterations.
        - confidence_threshold: Minimum confidence required to add a pseudo-labeled sample.
        """
        self.base_estimator_1 = base_estimator_1
        self.base_estimator_2 = base_estimator_2
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold

    def fit(self, X_1, X_2, y):
        """
        Fit the Co-Training classifier.

        Parameters:
        - X_1: Feature set for the first view (numpy array or pandas DataFrame).
        - X_2: Feature set for the second view (numpy array or pandas DataFrame).
        - y: Target labels, where -1 represents unlabeled data.

        Returns:
        - self: Fitted CoTrainingClassifier instance.
        """
        # Check that X_1 and X_2 have the same number of samples and match y
        X_1, y = check_X_y(X_1, y)
        X_2, y = check_X_y(X_2, y)
        assert X_1.shape[0] == X_2.shape[0], "X_1 and X_2 must have the same number of samples."

        self.classes_ = np.unique(y[y != -1])  # Exclude unlabeled samples (-1)

        self.base_estimator_1_ = clone(self.base_estimator_1)
        self.base_estimator_2_ = clone(self.base_estimator_2)

        labeled_mask = y != -1
        unlabeled_mask = ~labeled_mask

        X_1_labeled, y_labeled = X_1[labeled_mask], y[labeled_mask]
        X_2_labeled = X_2[labeled_mask]
        X_1_unlabeled, X_2_unlabeled = X_1[unlabeled_mask], X_2[unlabeled_mask]

        # Train initial models on labeled data
        self.base_estimator_1_.fit(X_1_labeled, y_labeled)
        self.base_estimator_2_.fit(X_2_labeled, y_labeled)

        for iteration in range(self.max_iter):
            # Predict probabilities on the unlabeled data
            probs_1 = self.base_estimator_1_.predict_proba(X_1_unlabeled)
            probs_2 = self.base_estimator_2_.predict_proba(X_2_unlabeled)

            # Identify high-confidence predictions for each view
            confident_1 = np.max(probs_1, axis=1) > self.confidence_threshold
            confident_2 = np.max(probs_2, axis=1) > self.confidence_threshold

            if not np.any(confident_1) and not np.any(confident_2):
                break  # Stop if no confident predictions

            # Add pseudo-labels to the opposite view
            pseudo_labels_1 = np.argmax(probs_1[confident_1], axis=1)
            pseudo_labels_2 = np.argmax(probs_2[confident_2], axis=1)

            if np.any(confident_1):
                self.base_estimator_2_.fit(
                    np.vstack([X_2_labeled, X_2_unlabeled[confident_1]]),
                    np.hstack([y_labeled, pseudo_labels_1])
                )

            if np.any(confident_2):
                self.base_estimator_1_.fit(
                    np.vstack([X_1_labeled, X_1_unlabeled[confident_2]]),
                    np.hstack([y_labeled, pseudo_labels_2])
                )

            # Update labeled and unlabeled sets
            X_1_unlabeled = X_1_unlabeled[~confident_1]
            X_2_unlabeled = X_2_unlabeled[~confident_2]

        return self

    def predict(self, X_1, X_2):
        """
        Predict class labels for samples in X_1 and X_2.

        Parameters:
        - X_1: Feature set for the first view.
        - X_2: Feature set for the second view.

        Returns:
        - y_pred: Predicted class labels.
        """
        check_is_fitted(self, ["base_estimator_1_", "base_estimator_2_"])

        X_1 = check_array(X_1)
        X_2 = check_array(X_2)
        
        # Combine predictions from both views
        preds_1 = self.base_estimator_1_.predict(X_1)
        preds_2 = self.base_estimator_2_.predict(X_2)

        # Use majority vote for final prediction
        y_pred = np.where(preds_1 == preds_2, preds_1, -1)  # -1 for conflicting predictions
        return y_pred

    def predict_proba(self, X_1, X_2):
        """
        Predict class probabilities for samples in X_1 and X_2.

        Parameters:
        - X_1: Feature set for the first view.
        - X_2: Feature set for the second view.

        Returns:
        - probas: Predicted class probabilities.
        """
        check_is_fitted(self, ["base_estimator_1_", "base_estimator_2_"])

        X_1 = check_array(X_1)
        X_2 = check_array(X_2)

        probs_1 = self.base_estimator_1_.predict_proba(X_1)
        probs_2 = self.base_estimator_2_.predict_proba(X_2)

        # Average probabilities from both views
        return (probs_1 + probs_2) / 2
