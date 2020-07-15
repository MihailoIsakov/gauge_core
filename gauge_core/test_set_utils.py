"""
Functions for building the test set given the dataset DataFrame, 
(and possibly the clusterer)
"""
import numpy as np
import sklearn

from . import clustering 


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

