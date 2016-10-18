import numpy as np

from sklearn.cross_validation import KFold


def get_partition(n_features, max_features_in_subset, random_state):
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


def get_bootstrapping(n_samples, n_subsets, samples_fraction, random_state):
    np.random.seed(random_state)

    for _ in range(n_subsets):
        yield np.unique(np.random.choice(a=np.arange(n_samples),
                                         size=int(samples_fraction * n_samples)))
