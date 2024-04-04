import numpy as np
import networkx as nx
import community as community_louvain
from collections import Counter
import pandas as pd
from joblib import Parallel, delayed
from pyunicorn.timeseries import RecurrenceNetwork

def get_visibility_graph(ts):
    g = NaturalVG(directed=None).build(ts)
    nxg = g.as_networkx()
    return nxg

def compute_basic_graph_measures(G):
    measures = {
        'number_of_nodes': nx.number_of_nodes(G),
        'number_of_edges': nx.number_of_edges(G),
        'average_degree': 2 * nx.number_of_edges(G) / nx.number_of_nodes(G),
        'density': nx.density(G),
        'average_clustering_coefficient': nx.average_clustering(G)
    }
    # The following measures are only defined for connected graphs
    if nx.is_connected(G):
        measures['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        measures['diameter'] = nx.diameter(G)
    else:
        measures['average_shortest_path_length'] = float('inf')  
        measures['diameter'] = float('inf') 
    
    measures['assortativity_coefficient'] = nx.degree_assortativity_coefficient(G)
    
    return measures

def calculate_network_measures(window, i):
    # Access the recurrence matrix from the window
    recurrence_matrix = window.recurrence_matrix() 

    # Convert the recurrence matrix to a NetworkX graph (adjacency matrix)
    G = nx.from_numpy_array(recurrence_matrix)
    
    # Calculate network measures
    measures = {
        'window': i,
        'transitivity': nx.transitivity(G),
        'average_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,
        'global_efficiency': nx.global_efficiency(G),
        'assortativity': nx.degree_assortativity_coefficient(G),
        'number_of_edges': nx.number_of_edges(G),
        'density': nx.density(G),
        'average_clustering_coefficient': nx.average_clustering(G),
        'average_degree': np.mean([degree for node, degree in G.degree()]),
        'edge_density': nx.density(G),
        'diameter': nx.diameter(G) if nx.is_connected(G) else np.nan
    }
    
    return measures

# Function to compute measures in parallel for each window
def compute_measures_in_parallel(time_series, window_size, step_size):
    results = []
    
    # Define the sliding window
    for i in range(0, len(time_series) - window_size + 1, step_size):
        window = RecurrencePlot(time_series[i:i + window_size], dim=1, tau=1, recurrence_rate=0.1, metric='supremum', silence_level=2)
        
        results.append(window)
    
    # Use Joblib to compute the measures in parallel
    parallel_results = Parallel(n_jobs=-1)(delayed(calculate_network_measures)(window, i) for i, window in enumerate(results))
    
    return pd.DataFrame(parallel_results)

def get_degree_distribution(g):
    
    # Assuming 'g' is a NetworkX graph object
    degrees = [val for (node, val) in g.degree()]

    # Create a degree distribution manually
    degree_distribution = {}
    for degree in degrees:
        if degree in degree_distribution:
            degree_distribution[degree] += 1
        else:
            degree_distribution[degree] = 1

    # Normalize the distribution by the number of nodes
    for degree in degree_distribution:
        degree_distribution[degree] /= float(g.number_of_nodes())

    # Store degrees and the probabilities in separate lists for plotting or other purposes
    ks = list(degree_distribution.keys())
    ps = list(degree_distribution.values())
    
    return ks, ps

def get_recurrence_network(recurrence_matrix, seed=42):
    np.random.seed(seed)  
    
    # Construct the recurrence network
    G = nx.Graph()

    # Iterate through the recurrence matrix to find recurrences
    # considering all elements without making the matrix symmetric
    rows, cols = np.where(recurrence_matrix == 1)

    # Create edges from the recurrences
    # filtering out the diagonal elements to avoid self-loops
    edges = [(row, col) for row, col in zip(rows, cols) if row != col]

    # Add all edges in bulk
    G.add_edges_from(edges)

    return G

def get_graph_measures(graph, seed=42):
        # Detect communities
    partition = community_louvain.best_partition(graph)

    # Compute the modularity score
    modularity_score = community_louvain.modularity(partition, graph)
    print(f"Modularity Score: {modularity_score}")
    
    # Community size distribution
    community_sizes = Counter(partition.values())

    # for community, size in community_sizes.items():
    #     print(f"Community {community}: {size} nodes")
    print(f"Number of communities: {len(community_sizes)}")
        
    # intra vs. inter community edges
    intra_community_edges = 0
    inter_community_edges = 0

    for (u, v) in graph.edges():
        if partition[u] == partition[v]:
            intra_community_edges += 1
        else:
            inter_community_edges += 1

    print(f"Intra-community edges: {intra_community_edges}")
    print(f"Inter-community edges: {inter_community_edges}")

    # evaluate role of nodes within communities
    # Calculate centrality measures for nodes within each community
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    
    print(f"Degree centrality: {degree_centrality}")
    print(f"Betweenness centrality: {betweenness_centrality}")
    print(f"Closeness centrality: {closeness_centrality}")

    # degree centrality for each node
    # for node, centrality in degree_centrality.items():
    #     print(f"Node {node}: Degree Centrality: {centrality}")
    
    # examine community cohesion and separation
    # Community cohesion using average clustering coefficient
    avg_clustering_per_community = {community: nx.average_clustering(graph.subgraph([node for node in graph if partition[node] == community]))
                                    for community in set(partition.values())}

    # for community, clustering in avg_clustering_per_community.items():
    #     print(f"Community {community}: Average Clustering Coefficient: {clustering}")
    
def get_nodes(graph, seed=None, aspect='equal', scale_nodes=1):
    # Generate positions for various layouts
    position = {
        'spring': nx.spring_layout(graph, seed=seed),
        'shell': nx.shell_layout(graph),
        'spiral': nx.spiral_layout(graph),
        'spectral': nx.spectral_layout(graph),
        'circular': nx.circular_layout(graph),
        'kamada_kawai': nx.kamada_kawai_layout(graph),
        'random': nx.random_layout(graph, seed=seed)
    }
    
    # Rescale positions if aspect set to equal
    if aspect == 'equal':
        for layout, pos in position.items():
            position[layout] = nx.rescale_layout_dict(pos)
    
    # Calculate degrees for each node
    degrees = dict(graph.degree())
    values = list(degrees.values())
    
    # Generate color and size maps based on degrees
    color_map = [value for value in values]
    size_map = [value * 1 for value in values]  # Adjust the multiplier for node sizes
    
    # Assemble the nodes dictionary
    nodes = {
        'position': position,
        'degrees': values,
        'sizes': size_map,
        'colors': color_map
    }
    
    return nodes
    
import json

def save_nodes(nodes, filename):
   # Need to convert numpy int64 keys to Python int and float32 to Python float
    for layout, pos in nodes['position'].items():
        nodes['position'][layout] = {int(k): [float(coord) for coord in v] for k, v in pos.items()}
    
    with open(filename, 'w') as file:
        json.dump(nodes, file, indent=4)

def load_nodes(filename):
    with open(filename, 'r') as file:
        nodes = json.load(file)
    # Convert positions back to the correct format if necessary
    for layout, pos in nodes['position'].items():
        for key, value in pos.items():
            pos[key] = tuple(value) 
    return nodes

def compare_communities(g):
    communities_modularity = nx.community.greedy_modularity_communities(g)
    communities_modularity = list(communities_modularity)

    # Assuming G is a NetworkX graph
    communities_label_propagation = nx.community.label_propagation_communities(g)
    communities_label_propagation = list(communities_label_propagation)


    for communities, algorithm in zip([communities_modularity, communities_label_propagation], ['greedy modularity', 'label propagation']):
        # contains the nodes that belong to one community
        community_of_node = {}
        for community_index, community in enumerate(communities):
            for node in community:
                community_of_node[node] = community_index

        # Initialize the count of inter-community edges
        inter_community_edges_count = 0

        # Iterate through each edge in the graph
        for u, v in g.edges():
            # Check if nodes connected by the edge belong to different communities
            if community_of_node[u] != community_of_node[v]:
                # If they belong to different communities, it's an inter-community edge
                inter_community_edges_count += 1

        print(f"Number of inter-community {algorithm} edges: {inter_community_edges_count}")

def create_transition_matrix(G):
    # Get the adjacency matrix (as a numpy matrix)
    adj_matrix = nx.to_numpy_array(G)
    
    # Normalize the matrix so each row sums to 1
    transition_matrix = adj_matrix / adj_matrix.sum(axis=1)
    
    # Replace NaNs with 0 (in case of nodes with no out-edges)
    transition_matrix = np.nan_to_num(transition_matrix)
    
    return transition_matrix

# Assuming communities is a list of sets, where each set contains nodes that form a community
def create_community_transition_matrices(communities):
    community_transition_matrices = {}
    for i, community in enumerate(communities):
        # Cceate a subgraph for the community
        H = G.subgraph(community)
        
        # store the transition matrix for the community
        community_transition_matrices[f'community_{i}'] = create_transition_matrix(H)
        
    return community_transition_matrices

def plot_diff_transition_matrix(community_transition_matrices, global_transition_matrix):
    
    ## still working on 
    
    # Normalize matrices (if they are not already probabilities)
    # Make sure rows sum to 1
    community_transition_matrices = {k: v / v.sum(axis=1, keepdims=True) for k, v in community_transition_matrices.items()}
    global_transition_matrix = global_transition_matrix / global_transition_matrix.sum(axis=1, keepdims=True)

    # Compute the difference between community transition matrices and the global transition matrix
    community_differences = {k: v - global_transition_matrix for k, v in community_transition_matrices.items()}

    # Plot the differences
    for community, difference_matrix in community_differences.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(difference_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Difference in Transition Probabilities: {community} vs Global')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.show()
