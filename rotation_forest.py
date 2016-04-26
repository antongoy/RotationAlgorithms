import numpy as np

from scipy.sparse import issparse

from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from sklearn.utils import check_array, check_random_state
from sklearn.utils.multiclass import check_classification_targets

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold

from joblib import Parallel, delayed


__author__ = 'anton-goy'


MAX_INT = np.iinfo(np.int32).max

def _get_partition(n_features, max_features_in_subset, random_state):    
    if n_features == max_features_in_subset:
        yield np.arange(n_features)
    else:
        n_subsets = n_features // max_features_in_subset

        if n_subsets == 1:
            n_subsets += 1

        cv_partitioner = KFold(n=n_features, 
                               n_folds=n_subsets,
                               shuffle=True, 
                               random_state=random_state)
        for _, test_index in cv_partitioner:
            yield test_index
        
def _get_bootstrapping(n_samples, n_subsets, samples_fraction, random_state):
    np.random.seed(random_state)

    for _ in range(n_subsets):
        yield np.unique(np.random.choice(a=np.arange(n_samples), 
                                         size=int(samples_fraction * n_samples)))
        
        
def _parallel_helper(obj, methodname, *args, **kwargs):
    """Private helper to workaround Python 2 pickle limitations"""
    return getattr(obj, methodname)(*args, **kwargs)
        
        
def _parallel_pca(rotation_matrix, partition_iterator, boostrapping_iterator, X, matrix_idx, n_matrices, verbose=0):
    if verbose > 1:
        print("applying rotation transformation %d of %d" % (matrix_idx + 1, n_matrices))
        
    for column_inds, row_inds in zip(partition_iterator, boostrapping_iterator):
        Xi = X[row_inds[:, None], column_inds]
        
        # Principal components are row-vectors
        Xi_components = PCA().fit(Xi).components_.T
        
        rotation_matrix[column_inds[:, None], column_inds] = Xi_components
        
        
    return rotation_matrix
        
        
def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    np.random.seed(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices

def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score fuction."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices

def _parallel_build_trees(rotation_matrix, tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None):
    """Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices)

        tree.fit(X.dot(rotation_matrix), y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X.dot(rotation_matrix), y, sample_weight=sample_weight, check_input=False)

    return tree



class RotationForestClassifier(BaseEnsemble):
    """
        Rotation Forest implementation based on "Rotation Forest: A New Classifier Ensemble Method" paper
        from Rodriguez et al. in Python.

        Parameters
        ----------
        n_estimators: integer, optional (default=10)
            The number of trees in the forest.

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
            
        bootstrap : boolean, optional (default=True)
            Whether bootstrap samples are used when building trees.
            
        oob_score : bool
            Whether to use out-of-bag samples to estimate
            the generalization error.
            
        n_jobs : integer, optional (default=1)
            The number of jobs to run in parallel for both `fit` and `predict`.
            If -1, then the number of jobs is set to the number of cores.
            
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
            
        verbose : int, optional (default=0)
            Controls the verbosity of the tree building process.
            
        warm_start : bool, optional (default=False)
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest.
            
        class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
        
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.
            
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``
            
            The "balanced_subsample" mode is the same as "balanced" except that weights are
            computed based on the bootstrap sample for every tree grown.
            
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
            
        Attributes
        ----------
        estimators_ : list of DecisionTreeClassifier
            The collection of fitted sub-estimators.
            
        rotation_matrices_: list of arrays of shape = [n_features, n_features]
            The collection of rotation matrices for fitted sub-estimators.
            
        classes_ : array of shape = [n_classes] or a list of such arrays
            The classes labels (single output problem), or a list of arrays of
            class labels (multi-output problem).
            
        n_classes_ : int or list
            The number of classes (single output problem), or a list containing the
            number of classes for each output (multi-output problem).
            
        n_features_ : int
            The number of features when ``fit`` is performed.
            
        n_outputs_ : int
            The number of outputs when ``fit`` is performed.
            
        feature_importances_ : array of shape = [n_features]
            The feature importances (the higher, the more important the feature).
            
        oob_score_ : float
            Score of the training dataset obtained using an out-of-bag estimate.
            
        oob_decision_function_ : array of shape = [n_samples, n_classes]
            Decision function computed with out-of-bag estimate on the training
            set. If n_estimators is small it might be possible that a data point
            was never left out during the bootstrap. In this case,
            `oob_decision_function_` might contain NaN.
    """
    def __init__(self, 
                 n_estimators=10,
                 max_features_in_subset=3,
                 samples_fraction=0.75,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RotationForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf", 
                              "max_leaf_nodes", "random_state"))
        
        self.max_features_in_subset = max_features_in_subset
        self.samples_fraction = samples_fraction
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap=bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        
        
    def fit(self, X, y, sample_weight=None):
        """Build a forest of rotation trees from the training set (X, y).
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        
        if issparse(X):
            X.sort_indices()
            
        n_samples, self.n_features_ = X.shape
        
        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        
        y, expanded_class_weight = self._validate_y_class_weight(y)
        
        if getattr(y, "dtype", None) != np.float64 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)
            
        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
        
        self._validate_estimator()
        
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.max_features_in_subset > self.n_features_:
            raise ValueError("max_features_in_subset=%d must be smaller than"
                             " n_features=%d" 
                             % (self.max_features_in_subset, self.n_features_))
        
        random_state = check_random_state(self.random_state)
        
        if not self.warm_start:
            self.estimators_ = []
            self.rotation_matrices_ = []
            self.partitions_ = []
            self.bootstrappings_ = []
            
        n_more_estimators = self.n_estimators - len(self.estimators_)
        
        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            rotation_matrices = []
            partitions = []
            bootstrappings = []
            
            for i in range(n_more_estimators):
                tree = self._make_estimator(append=False)
                tree.set_params(random_state=random_state.randint(MAX_INT))
                partition_iterator = _get_partition(self.n_features_, 
                                                    self.max_features_in_subset, 
                                                    random_state.randint(MAX_INT))
                
                bootstrapping_iterator = _get_bootstrapping(n_samples, 
                                                            self.n_features_ // self.max_features_in_subset, 
                                                            self.samples_fraction, 
                                                            random_state.randint(MAX_INT))
                trees.append(tree)
                partitions.append(partition_iterator)
                bootstrappings.append(bootstrapping_iterator)
                rotation_matrices.append(np.zeros((self.n_features_, self.n_features_), dtype=np.float32))
                
            rotation_matrices = Parallel(n_jobs=self.n_jobs, 
                                         verbose=self.verbose,
                                         backend="threading")(
                delayed(_parallel_pca)(r, p, b, X, i, len(rotation_matrices), verbose=self.verbose) 
                                       for i, (r, p, b) in enumerate(zip(rotation_matrices, 
                                                                         partitions, 
                                                                         bootstrappings)))

            trees = Parallel(n_jobs=self.n_jobs, 
                             verbose=self.verbose,
                             backend="threading")(
                delayed(_parallel_build_trees)(r, t, self, X, y, sample_weight, i, len(trees),
                                               verbose=self.verbose, class_weight=self.class_weight)
                                               for i, (t, r) in enumerate(zip(trees, rotation_matrices)))

            self.estimators_.extend(trees)
            self.rotation_matrices_.extend(rotation_matrices)
            self.partitions_.extend(partitions)
            self.bootstrappings_.extend(bootstrappings)
            
        if self.oob_score:
            self._set_oob_score(X, y)
            
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            
        return self
    
    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return self.estimators_[0]._validate_X_predict(X, check_input=True)
    
    
    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before `feature_importances_`.")

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   backend="threading")(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_)

        return sum(all_importances) / len(self.estimators_)
    
    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=np.float32, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = []

        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))

        for estimator, rotation_matrix in zip(self.estimators_, self.rotation_matrices_):
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :].dot(rotation_matrix),
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_
        
    
    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('auto', 'balanced', 'subsample', 'balanced_subsample')
            if isinstance(self.class_weight, six.string_types):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample". Given "%s".'
                                     % self.class_weight)
                if self.class_weight == "subsample":
                    warn("class_weight='subsample' is deprecated in 0.17 and"
                         "will be removed in 0.19. It was replaced by "
                         "class_weight='balanced_subsample' using the balanced"
                         "strategy.", DeprecationWarning)
                if self.warm_start:
                    warn('class_weight presets "balanced" or "balanced_subsample" are '
                         'not recommended for warm_start if the fitted data '
                         'differs from the full dataset. In order to use '
                         '"balanced" weights, use compute_class_weight("balanced", '
                         'classes, y). In place of y you can use a large '
                         'enough sample of the full training set target to '
                         'properly estimate the class frequency '
                         'distributions. Pass the resulting weights as the '
                         'class_weight parameter.')

            if (self.class_weight not in ['subsample', 'balanced_subsample'] or
                    not self.bootstrap):
                if self.class_weight == 'subsample':
                    class_weight = 'auto'
                elif self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                with warnings.catch_warnings():
                    if class_weight == "auto":
                        warnings.simplefilter('ignore', DeprecationWarning)
                    expanded_class_weight = compute_sample_weight(class_weight,
                                                                  y_original)

        return y, expanded_class_weight
    
    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is a vote by the rotation trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the rotation trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             backend="threading")(
            delayed(_parallel_helper)(e, 'predict_proba', X.dot(r),
                                      check_input=False)
            for e, r in zip(self.estimators_, self.rotation_matrices_))

        # Reduce
        proba = all_proba[0]

        if self.n_outputs_ == 1:
            for j in range(1, len(all_proba)):
                proba += all_proba[j]

            proba /= len(self.estimators_)

        else:
            for j in range(1, len(all_proba)):
                for k in range(self.n_outputs_):
                    proba[k] += all_proba[j][k]

            for k in range(self.n_outputs_):
                proba[k] /= self.n_estimators

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba