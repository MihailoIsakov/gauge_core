import numpy as np
import networkx as nx
import logging


def find_cluster_members(G, node_idx, epsilon):
    """
    Given a minimum spanning tree G from HDBSCAN, the function finds all elements that belong to the 
    same cluster as the node with index node_idx, if we ran DBSCAN with the provided epsilon parameter.
    """

    # First, keep going up the tree until the inverse of lambda (edge weight) is smaller than the epsilon
    child = node_idx
    parent = None
    while True:
        if len(list(G.predecessors(child))) == 0:
            break

        parent = list(G.predecessors(child))[0]
        if 1 / G.edges[parent, child]['weight'] < epsilon:
            child = parent
            continue
        else: 
            break

    # Now that we have the parent, extract all the parent and all the children as a subgraph
    nodes = set([parent])

    search = nx.dfs_successors(G, parent)
    for value in search.values():
        nodes = nodes.union(value) 

    return nodes


def forest_to_tree_nodes(G):
    """
    A forest is a set of unconnected trees. This function returns a list of sets, where each set
    contains the nodes of a single tree.
    """
    connected_components = list(nx.connected_components(G.to_undirected()))

    component_sizes = []
    for comp in connected_components: 
        component_sizes.append(len(comp)) 

    logging.info("Found {} clusters. {} clusters containt a single outlier, out of {} total points".format(
        len(connected_components), np.sum(np.array(component_sizes) == 1), np.sum(component_sizes)))

    return connected_components


def HDBSCAN_to_DBSCAN(clusterer, epsilon):
    """
    Since running DBSCAN is slow, we reuse the existing minimum spanning tree provided from 
    HDBSCAN and remove any edges that are longer than epsilon. Whatever disjoint graphs are
    left form the clusters DBSCAN would have returned.

    Assumes that the nodes in the graph are labeled from 0 to the number of nodes - 1

    Args:
        clusterer: HDBSCAN's clusterer object
        epsilon: DBSCAN epsilon parameter

    Returns:
        list of clusters, where each cluster is a list of row indexes 
    """
    G = clusterer.condensed_tree_.to_networkx()

    logging.info("Running DBSCAN with eps={} on the HDBSCAN minimum spanning tree".format(epsilon))

    # First, delete any edges that with 1 / weights larger than epsilon
    for edge in list(G.edges):
        if 1 / G[edge[0]][edge[1]]['weight'] > epsilon:
            G.remove_edge(*edge)

    clusters = forest_to_tree_nodes(G)

    # Since the graph contains some branch nodes, and not just leaf nodes, remove those branch nodes
    branch_nodes = range(len(clusterer.labels_) + 1, np.max(G.nodes()) + 1)
    for idx in range(len(clusters)):
        clusters[idx] = clusters[idx].difference(branch_nodes)

    # This will leave some empty clusters. Let's remove them
    clusters = [c for c in clusters if len(c) > 0]

    return clusters

