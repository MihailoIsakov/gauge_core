import os
import sys
import numpy as np
import networkx as nx
from joblib import Memory


memory = Memory(os.path.join(os.path.dirname(__file__), ".cache"))


new_node_idx = 0


def split_node(G, parent):
    """
    Takes a node with 3 or more edges, splits it into multiple nodes of degree 2.
    E.g.: 

    G                G
  g |   w          g |   w
    P-------         M------
    |\     |       w |     |
    | \    |   --\   P---  |
    |  \   |   --/   |  |  |
    A   B  C         A  B  C
    """
    global new_node_idx

    while G.out_degree(parent) > 2: 
        try: 
            grandparent = list(G.predecessors(parent))[0]
        except:
            return 
        connected = list(G[parent]) 
        weights   = [G[parent][c]['weight'] for c in connected]
        child     = connected[np.argmin(weights)] # first child 

        # Now delete the grandparent-parend and parent-child edges
        gp_weight = G[grandparent][parent]['weight']
        pc_weight = G[parent][child]['weight']
        G.remove_edges_from([(grandparent, parent), (parent, child)])

        # Now add middle, and connect G-M, M-P, M-C
        middle = "{}-{}".format(str(grandparent).split('-')[0], new_node_idx)
        new_node_idx += 1
        assert middle not in G
        G.add_node(middle)
        G.add_weighted_edges_from([(grandparent, middle, gp_weight), (middle, parent, pc_weight), (middle, child, pc_weight)])


def split_multidegree_nodes(G):
    """
    Takes a graph G and splits any nodes that have more than 2 edges leading out from it.
    The edges with the largest lambdas should be processed first.
    """
    # while np.max([v for k, v in G.out_degree]) > 2: 
        # for parent in list(G.nodes):
            # split_node(G, parent)

    # assert np.max([v for k, v in G.out_degree]) <= 2

    # since the inserted nodes don't have sizes, this should populate them
    root = [n for n, d in G.in_degree() if d==0][0]
    _populate_node_sizes(G, root)


def get_leaves(G): 
    return [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]


def _populate_node_sizes(G, node):
    """
    Nodes in G should have sizes, but some don't after we split multidegree nodes. This
    recursively recalculates sizes of all nodes in the graph starting from node.
    """
    children = list(G[node])
    # if a leaf node, set size to 1
    if len(children) == 0:
        size = 1
    # branches add up children's sizes 
    # elif len(children) == 2:
        # size = _populate_node_sizes(G, children[0]) + _populate_node_sizes(G, children[1])
    # else:
        # raise RuntimeError("Node {} has {} children.".format(node, len(children)))
    else:
        size = sum([_populate_node_sizes(G, children[c]) for c in range(len(children))])

    G.nodes[node]['size'] = size

    return size


def _populate_users_and_apps(df, G, node):
    """
    Populates nodes in G with sets of users and apps that they contain.
    """
    children = list(G[node])
    if len(children) == 0:
        apps = set([df.iloc[node]['apps_short']])
        users = set([df.iloc[node]['users']])
    else: 
        apps, users = set(), set()
        for c in range(len(children)):
            a, u = _populate_users_and_apps(df, G, children[c])
            apps = apps.union(a)
            users = users.union(u)
        
    G.nodes[node]['apps'] = apps
    G.nodes[node]['users'] = users

    return apps, users


def _merge_chain_nodes(G, node, dont_merge=[]):
    """
    Traverses the graph, and converts chains such as A-B-C to A-C or A-B to B.
    Dont merge specifies nodes that should not be merged even if part of chain
    """
    if len(list(G.predecessors(node))) == 1 and len(list(G.successors(node))) == 1 and node not in dont_merge:
        parent = list(G.predecessors(node))[0]
        child  = list(G.successors  (node))[0]  # noqa: E211
        weight = G[node][child]['weight']

        # G.remove_edges_from([(parent, node), (node, child)])
        G.remove_node(node)
        G.add_weighted_edges_from([(parent, child, weight)])

        _merge_chain_nodes(G, child)
    else:
        for child in list(G.successors(node)):
            _merge_chain_nodes(G, child)


def build_condensed_graph(G, min_epsilon, min_cluster_size, dont_merge=[]):
    """ 
    Finds nodes in the graph that have edges weight weights above min_epsilon,
    and both children have a size larger than min_cluster_size.
    """
    def filter_node(n):
        return G.nodes[n]['size'] > min_cluster_size

    def filter_edge(n1, n2):
        return 1 / G[n1][n2]['weight'] > min_epsilon

    SG = nx.subgraph_view(G, filter_node=filter_node, filter_edge=filter_edge)
    SG = nx.DiGraph(SG)

    root = [n for n, d in SG.in_degree() if d==0][0]
    _merge_chain_nodes(SG, root, dont_merge=dont_merge)

    # Remove orphans
    SG.remove_nodes_from(list(nx.isolates(SG)))

    return SG


def tree_layout(G, min_yaxis_spacing=0.5):

    def dfs_assignment(G, node, pos, next_x, min_yaxis_spacing): 
        """
        Calculates the node's position and recursively calculates it's childrens positions.
        The y position is calculated from epsilon, while the x position is calculated by first
        assigning leaves integer positions, and the branches take the average of their children.
        """
        parent = list(G.predecessors(node))
        children = list(G.successors(node))

        # Calculate X positon
        if len(children) == 0: 
            x = next_x
            next_x += 1
        else: 
            # Get children to assign their X's, and take their mean
            for child in children: 
                pos, next_x = dfs_assignment(G, child, pos, next_x, min_yaxis_spacing=min_yaxis_spacing)
            x = np.mean([pos[child][0] for child in children])

        # Calculate Y position
        if len(parent) == 1:
            y = 1 / G[parent[0]][node]['weight']
        else:
            y = 12

        pos[node] = [x, y]

        # Space out nodes on y axis
        if len(children) >= 1:
            top_child = np.max([pos[child][1] for child in children])
            if pos[node][1] - top_child < min_yaxis_spacing:
                pos[node][1] = top_child + min_yaxis_spacing

        return pos, next_x

    root = [n for n, d in G.in_degree() if d==0][0]
    pos = {}
    pos, _ = dfs_assignment(G, root, pos, 0, min_yaxis_spacing=min_yaxis_spacing)

    return pos


@memory.cache
def build_hierarchy(df, clusterer, min_cluster_size=1000):
    """
    Returns a dictionary with a single field nodes, which contains 
    a list of node objects. Each element of the list contains the fields
    x, y (positions constructed by the layout mechanism), node index/name,
    size, epsilon (same as y for now), parent (index of parent node), 
    children (0 to 2 indexes of children nodes).

    Args:
        clusterer: an HDBSCAN object

    Returns: 
        a dictionary where with a single 'nodes' key, which holds a list of 
        dictionaries, each describing a cluster using several features. 
    """
    ct = clusterer.condensed_tree_
    G  = ct.to_networkx()

    sys.setrecursionlimit(10000)
    split_multidegree_nodes(G)

    # Populate users and apps in hierarchy
    root = [n for n, d in G.in_degree() if d==0][0]
    _populate_users_and_apps(df, G, root)

    CG = build_condensed_graph(G, 1., min_cluster_size=min_cluster_size)
    coord = tree_layout(CG)
    sizes = dict(nx.get_node_attributes(CG, 'size').items())

    hierarchy = {"nodes": []}
    for node in CG:
        node_object = {
            "index"   : str(node),
            "x"       : float(coord[node][0]),
            "y"       : coord[node][1],
            "epsilon" : coord[node][1],
            "size"    : sizes[node],
            "parent"  : str(list(CG.predecessors(node))[0]) if len(list(CG.predecessors(node))) > 0 else None, 
            "children": list([str(x) for x in CG.successors(node)]),
            "users"   : list(CG.nodes[node]['users']),
            "apps"    : list(CG.nodes[node]['apps']),
        }

        hierarchy['nodes'].append(node_object)

    return hierarchy


def build_nested_hierarchy(df, clusterer, min_cluster_size=1000):
    """
    """
    tree_list = build_hierarchy(df, clusterer, min_cluster_size)['nodes']
    root_index = [node['index'] for node in tree_list if node['parent'] is None][0]

    def nest(index):
        node = [node for node in tree_list if node['index'] == index][0]

        # Recursively nest children
        if len(node['children']) > 0:
            node['children'] = [nest(c) for c in node['children']]

        return node

    return nest(root_index)




