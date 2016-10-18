import numpy as np

from numpy.core.umath_tests import inner1d

from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import is_regressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted

from .misc_functions import get_partition, get_bootstrapping

MAX_INT = np.iinfo(np.int32).max


class RotationAdaBoostClassifier(BaseEnsemble):
    """
    A Rotation AdaBoost classifier.
    
    On each step of Adaboost classifier a weighted rotation is perfomed.
    
    Parameters
    ----------
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
        
     max_features_in_subset: integer, optional (default=3)
            The number of feature subsets which are slitted on each step.

    samples_fraction: float, optional (default=0.75)
        The fraction of bootstrap samples to draw for each classifier in ensemble.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
        Note: this parameter is tree-specific.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
        Note: this parameter is tree-specific.
        
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
        
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
        
    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
        
    rotation_matrices_: list of arrays of shape = [n_features, n_features]
        The collection of rotation matrices for fitted sub-estimators.
            
    classes_ : array of shape = [n_classes]
        The classes labels.
        
    n_classes_ : int
        The number of classes.
        
    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.
        
    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.
        
    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.
    """
    
    def __init__(self, 
                 n_estimators=50,
                 criterion='gini',
                 max_features_in_subset=3,
                 samples_fraction=0.75,
                 learning_rate=1.,
                 min_samples_split=2,
                 min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.,
                 max_depth=3, 
                 max_leaf_nodes=None,
                 algorithm='SAMME.R',
                 verbose=0,
                 random_state=None):
        
        super(RotationAdaBoostClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf", 
                              "max_leaf_nodes", "random_state")) 
        self.learning_rate = learning_rate
        self.max_features_in_subset = max_features_in_subset
        self.samples_fraction = samples_fraction
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.algorithm = algorithm
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted classifier/regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
            
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
            
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
            Returns self.
        """
        
        _, self.n_features_ = X.shape
        
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
            
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)    
            
        if self.max_features_in_subset > self.n_features_:
            raise ValueError("max_features_in_subset=%d must be smaller than"
                             " n_features=%d" 
                             % (self.max_features_in_subset, self.n_features_)) 

        X, y = check_X_y(X, y, accept_sparse='csc', dtype=np.float32,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.rotation_matrices_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)
        
        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y, 
                sample_weight, 
                random_state.randint(MAX_INT))

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Implement a single boost.
        
        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.
        
        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
            
        y : array-like of shape = [n_samples]
            The target values (class labels).
            
        sample_weight : array-like of shape = [n_samples]
            The current sample weights.
            
        random_state : int
            A seed for random state generator
        
        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.
            
        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.
            
        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    def _make_rotation_matrix(self, X, sample_weight, random_state):
        """
        Make rotation matrix for current boost.
        
        Parameters
        ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        
        sample_weight : array-like of shape = [n_samples]
            The current sample weights.
        
        random_state : int
            A seed for random state generator
            
        Returns
        -------
        rotation_matrix : array of shape = [n_features, n_features]
            A rotation matrix for current iteration.
        """

        n_samples, n_features = X.shape
        rotation_matrix = np.zeros((n_features, n_features), dtype=np.float32)

        partition_iter = get_partition(n_features,
                                       self.max_features_in_subset,
                                       random_state)
        bootstrap_iter = get_bootstrapping(n_samples,
                                           n_features // self.max_features_in_subset,
                                           self.samples_fraction,
                                           random_state)

        for column_inds, row_inds in zip(partition_iter, bootstrap_iter):
            # Make Weighted PCA
            Xi = X[row_inds[:, None], column_inds] * np.sqrt(sample_weight[row_inds, None])

            # Principal components are row-vectors
            Xi_components = PCA().fit(Xi).components_.T

            rotation_matrix[column_inds[:, None], column_inds] = Xi_components
    
        self.rotation_matrices_.append(rotation_matrix)
        return rotation_matrix
        
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator()
        rotation_matrix = self._make_rotation_matrix(X, sample_weight, random_state)
        
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X.dot(rotation_matrix), y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X.dot(rotation_matrix))

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator()
        rotation_matrix = self._make_rotation_matrix(X, sample_weight, random_state)

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X.dot(rotation_matrix), y, sample_weight=sample_weight)

        y_predict = estimator.predict(X.dot(rotation_matrix))

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """
        Predict classes for X.
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
    
    def _samme_proba(self, estimator, rotation_matrix, n_classes, X):
        proba = estimator.predict_proba(X.dot(rotation_matrix))

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])
    
    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = check_array(X, accept_sparse='csr', dtype=np.float32)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(self._samme_proba(estimator, R, n_classes, X)
                       for estimator, R in zip(self.estimators_, self.rotation_matrices_))
        else:   # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X.dot(R)) == classes).T * w
                       for estimator, w, R in zip(self.estimators_,
                                               self.estimator_weights_, 
                                               self.rotation_matrices_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
    
    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")
    
    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = check_array(X, accept_sparse='csr', dtype=np.float32)

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba(estimator, R, n_classes, X)
                        for estimator, R in zip(self.estimators_, self.rotation_matrices_))
        else:   # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X.dot(R)) * w
                        for estimator, w, R in zip(self.estimators_,
                                                self.estimator_weights_, 
                                                self.rotation_matrices_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba