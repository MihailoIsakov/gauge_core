"""
Functions for building the test set given the dataset DataFrame, 
(and possibly the clusterer)
"""
from collections import Counter
import numpy as np
import sklearn
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils.validation import check_array

from gauge_core import clustering 


def random_split(df, y_column, keep_columns=None, test_size=0.2):
    """
    Original test set function that doesn't show generalization well.

    Args: 
        df: dataset DataFrame 
        y_column: name of column we want to use as target 
        keep_columns: either None, or a list of columns we want to keep in the input sets 

    Returns:
        training and test input and target data
    """
    if keep_columns is None:
        keep_columns = set(df.columns).difference([y_column])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[keep_columns], df[y_column], test_size=test_size)

    return X_train, X_test, y_train, y_test 


def per_app_split(df, app_name, y_column, keep_columns=None):
    """
    Selects all jobs of the given short application name, and uses them in the test set.

    Args: 
        df: dataset DataFrame 
        app_name: name of application we want to use as test set
        y_column: name of column we want to use as target 
        keep_columns: either None, or a list of columns we want to keep in the input sets 

    Returns:
        training and test input and target data
    """
    if keep_columns is None:
        keep_columns = set(df.columns).difference([y_column])

    X_train = df[df.apps_short != app_name][keep_columns]
    y_train = df[df.apps_short != app_name][y_column]
                                          
    X_test  = df[df.apps_short == app_name][keep_columns]
    y_test  = df[df.apps_short == app_name][y_column]

    return X_train, X_test, y_train, y_test 


def DBSCAN_based_split(df, clusterer, epsilon, y_column, keep_columns=None, test_size=0.2):
    """
    Clusters the dataset with a given epsilon, and selects a number of clusters 
    at random to put in the test set. Since we are using DBSCAN, we can guarantee
    a minimum distance of epsilon between any point in the training and test set.

    Args:
        df: dataset DataFrame
        clusterer: an HDBSCAN clusterer object 
        epsilon: epsilon distance at which to cluster using DBSCAN
        test_size: percentage of the total dataset to put in the test set

    Returns:
        training and test input and target data
    """
    clusters = clustering.HDBSCAN_to_DBSCAN(clusterer, epsilon)
    cluster_sizes = np.array([len(c) for c in clusters])

    test_rows = set() 

    # Hystersis - test set can be within 5% of the target size
    min_test_rows = 0.95 * df.shape[0] * test_size
    max_test_rows = 1.05 * df.shape[0] * test_size

    while len(test_rows) < min_test_rows:
        # Remove from consideration all clusters that would overfill our test set 
        max_cluster_size = np.maximum(0, max_test_rows - len(test_rows))
        cluster_sizes[cluster_sizes > max_cluster_size] = 0

        # Pick a cluster based with probability based on cluster size 
        chosen_idx = np.random.choice(range(len(cluster_sizes)), p=cluster_sizes / np.sum(cluster_sizes))

        # Remove the chosen cluster from further consideration
        cluster_sizes[chosen_idx] = 0

        # Add all jobs from that cluster to the test set 
        test_rows = test_rows.union(clusters[chosen_idx])

    # Subtract the test rows from the training rows, and convert both to lists
    training_rows = list(set(range(len(clusterer.labels_))).difference(test_rows))
    test_rows     = list(test_rows)

    X_train = df.iloc[training_rows][keep_columns]
    y_train = df.iloc[training_rows][y_column]

    X_test  = df.iloc[test_rows][keep_columns]
    y_test  = df.iloc[test_rows][y_column]

    return X_train, X_test, y_train, y_test 


class AppFold(LeaveOneGroupOut):
    """ 
    AppFold cross-validator 

    Provides train/test indices to split data in train/test sets. Each
    set of jobs belonging to one of the top n applications is used once
    as the test set (singleton) while the remaining samples form the 
    training set. Jobs that do not belong to the n most numerous 
    applications are never included in the test set.
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, X, y=None):
        """
        Creates the groups based on the DataFrame X and the n_splits parameter,
        then just calls the base class implementation.
        """
        top_apps = Counter(X.apps_short).most_common()[:self.n_splits]
        top_apps = [ta[0] for ta in top_apps]

        for idx, app in enumerate(top_apps):
            train_index = np.flatnonzero(X.apps_short != app)
            test_index  = np.flatnonzero(X.apps_short == app)

            yield train_index, test_index


class DBSCANFold(GroupKFold):

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, clusterer=None, epsilon=1):
        if clusterer is None:
            raise RuntimeError("Clusterer not provided")
        groups = np.zeros(len(X))

        clusters = clustering.HDBSCAN_to_DBSCAN(clusterer, epsilon=epsilon)
        groups = np.zeros(len(X))

        for idx, c in enumerate(clusters):
            groups[list(c)] = idx

        return super().split(X, y, groups)


if __name__ == "__main__":
    import gauge_core
    df, clusterer = gauge_core.dataset.default_dataset()

    top_apps = Counter(df.apps_short).most_common()[:5]
    print(top_apps)

    fold = AppFold(n_splits=5)
    for train_index, test_index in fold.split(df):
        print(len(train_index), len(test_index))
        # Make sure that only a single app exists in the test set 
        print(set(df.iloc[test_index].apps_short))
        
    fold = DBSCANFold(n_splits=5)
    for train_index, test_index in fold.split(df, clusterer=clusterer):
        print(len(train_index), len(test_index))
        # Make sure that only a single app exists in the test set 
        print(set(df.iloc[test_index].apps_short))
