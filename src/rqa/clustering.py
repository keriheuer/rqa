
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from sklearn.ensemble import IsolationForest
import community as community_louvain
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import infomap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from cdlib import TemporalClustering, algorithms, evaluation

from itertools import product
from scipy.optimize import linear_sum_assignment

### PARAMETER ESTIMATION ####

def elbow_method(data, plot=True):
    distortions = []
    K = range(1, 10)  # Adjust the range of k as needed
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    
    if plot:
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
    
    # Optional: systematically find the elbow point, the point of maximum curvature
    # note that this is an approximation and works well for clear elbow points
    k_optimal = np.argmax(np.diff(np.diff(distortions))) + 2  # +2 as the second derivative starts from k=2
    print(f"The optimal value of k using the Elbow Method is: {k_optimal}")
    return k_optimal

def silhouette_method(data, plot=True):
    silhouette_scores = []
    K = range(2, 10)  # Silhouette analysis requires at least 2 clusters
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        preds = kmeanModel.fit_predict(data)
        score = silhouette_score(data, preds)
        silhouette_scores.append(score)
    
    if plot:
        plt.figure(figsize=(16,8))
        plt.plot(K, silhouette_scores, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis showing the optimal k')
        plt.show()
    
    k_optimal = K[np.argmax(silhouette_scores)]
    print(f"The optimal value of k using the Silhouette Method is: {k_optimal}")
    return k_optimal

def estimate_dbscan_params(data):
    # Use NearestNeighbors to estimate a good value for eps
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Sort distances
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    eps = np.percentile(distances, 90)  # Taking 90th percentile as eps
    
    # min_samples estimation based on data heuristic
    min_samples = max(2, int(len(data) * 0.01))  # 1% of data points
    
    return eps, min_samples

def estimate_n_neighbors(data, mode):
    """
    Estimate n_neighbors based on the data characteristics.
    For network data, use the average community size or average degree.
    For time series or other data types, a default heuristic is used.
    """
    if mode in ['rn', 'vg']:  # For recurrence networks or visibility graphs
        # Assuming 'data' is a NetworkX graph
        avg_degree = np.mean([degree for node, degree in nx.degree(data)])
        estimated_n_neighbors = max(2, int(avg_degree))  # Ensure at least 2 neighbors
    else:  # Fallback for other modes
        estimated_n_neighbors = min(20, len(data) // 5)  # Example heuristic for non-graph data
    
    return estimated_n_neighbors

def estimate_k(data):
    distortions = []
    K_range = range(1, min(10, len(data) // 2))
    for k in K_range:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    
    # Estimate the elbow point as the optimal k; simplified approach
    k_opt = np.argmax(np.diff(np.diff(distortions))) + 2  # Placeholder
    return max(2, k_opt)  # Ensure at least 2 clusters

def iterative_dbscan_param_estimation(data):
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}
    for eps in np.linspace(0.1, 1.0, num=10):  # Example range for eps
        for min_samples in range(2, 10):  # Example range for min_samples
            dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan_model.fit_predict(data)
            if len(set(labels)) > 1:  # More than one cluster plus noise
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
    return best_params

#### PLOTTING FUNCTIONS ####

def plot_results(results, save=True, fname=None, dir="./"):
    method = results.get('method', 'unknown_method')
    mode = results.get('mode', 'unknown_mode')
    detection_type = list(results.keys())[-1]  # Assuming the last key is the detection type
    
    # Construct filename if not provided
    if not fname:
        fname = f"{mode}_{method}_{detection_type}.pdf"
    else:
        # Handle the case where fname is explicitly provided or needs to be derived from a list
        if isinstance(fname, list):
            print("Error: fname as a list is not supported for individual plot saving.")
            return
        fname = f"{fname}.pdf" if not fname.endswith('.pdf') else fname

    plt.figure()
    plt.title(f"{method} Results ({mode})")
    
    # REPLACE WITH actual plotting functionality
    ##############
    
    if save:
        full_path = os.path.join(dir, fname)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Plot saved to {full_path}")
    
    plt.show()

def plot_individual_result(results, save, fname, dir):
    plt.figure()
    plt.title(f"Results for {results['method']}")
    
    # Add plotting logic here based on the type of results (clusters, anomalies, communities)
    #############
    
    if save:
        plt.savefig(os.path.join(dir, fname))
    plt.show()

#### HELPER FUNCTIONS ####

def detect_communities_louvain(snapshot):
    """
    Detect communities in a network snapshot using the Louvain method.
    """
    partition = community_louvain.best_partition(snapshot)
    return partition

def align_communities(communities_list):
    """
    Align communities across snapshots based on overlap.
    This is a very simplistic approach for illustrative purposes.
    """
    aligned_communities = [communities_list[0]]  # Start with the first snapshot's communities

    for current_communities in communities_list[1:]:
        # Placeholder for alignment logic
        # need to compare community compositions and match them based on overlap
        aligned_communities.append(current_communities) 

    return aligned_communities

def calculate_jaccard_similarity(community_a, community_b):
    intersection = len(set(community_a) & set(community_b))
    union = len(set(community_a) | set(community_b))
    return intersection / union if union else 0

def advanced_alignment(communities_t, communities_t_plus_1, match='bipartite'):
    """
    Perform advanced alignment of communities between two time steps using Jaccard similarity.
    communities_t and communities_t_plus_1 are lists of sets, where each set represents a community.
    """
    # Calculate Jaccard similarity for all community pairs between t and t+1
    jaccard_scores = [[calculate_jaccard_similarity(community_a, community_b) for community_b in communities_t_plus_1] for community_a in communities_t]
    
    if match == 'bipartite':
        # Create a bipartite graph
        B = nx.Graph()
        B.add_nodes_from(range(len(communities_t)), bipartite=0)
        B.add_nodes_from(range(len(communities_t), len(communities_t) + len(communities_t_plus_1)), bipartite=1)
        B.add_weighted_edges_from([(i, j+len(communities_t), jaccard_scores[i][j-len(communities_t)]) for i, j in product(range(len(communities_t)), range(len(communities_t), len(communities_t) + len(communities_t_plus_1)))])
        
        # Perform maximum weight matching
        matched = nx.algorithms.bipartite.matching.maximum_weight_matching(B, maxcardinality=True)
        return matched
    elif match == 'linear':
        # Linear programming approach for maximum weight matching
        cost_matrix = -1 * np.array(jaccard_scores)  # Convert to cost for minimization
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))
    else:
        raise ValueError("Invalid match parameter. Use 'bipartite' or 'linear'.")

def map_labels(labels, data, mode):
    # Placeholder function. Logic to map labels to data based on mode
    # E.g., for an RP (recurrence plot), mapping might involve converting a 1D array of labels to the 2D format of the RP
    # need to consider different data structure and the algorithm's output
    mapped_data = data
    return mapped_data

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_results(filename):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

#### ALGORITHMS ####

def isolation_forest(data, mode, seed=None, plot=True):
    # Parameter estimation or adjustments based on mode
    # For simplicity, use default parameters, but could involve
    # dynamic estimation based on data characteristics
    clf = IsolationForest(random_state=seed)
    
    # Data preprocessing based on mode (if necessary)
    if mode in ['rp', 'rn', 'vg']:  # Example for handling non-time series data
        data_preprocessed = data.flatten() if mode == 'rp' else data  # Example adjustment
    else:  # 'ts' or 'all'
        data_preprocessed = data
    
    anomalies = clf.fit_predict(data_preprocessed)
    
    results = {
        'method': 'Isolation Forest',
        'seed': seed,
        'mode': mode,
        'anomalies': anomalies.tolist(),  # Ensure JSON serializability
        'param_estimation': 'default'  # Placeholder for parameter estimation details
    }
    
    if plot:
        plot_results(results)
    
    return results

def local_outlier_factor(data, mode, seed=None, n_neighbors=None, plot=True):
    """
    Apply Local Outlier Factor (LOF) for anomaly detection.
    If n_neighbors is None, estimate it based on the data properties.
    """
    if n_neighbors is None:
        n_neighbors = estimate_n_neighbors(data, mode)
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    # Fit and predict; handling varies if 'data' is a graph
    if mode in ['rn', 'vg']:
        # Convert graph to a suitable representation for LOF, e.g., adjacency matrix
        data_processed = nx.to_numpy_array(data)
        anomalies = lof.fit_predict(data_processed)
    else:
        anomalies = lof.fit_predict(data)
    
    results = {
        'method': 'Local Outlier Factor',
        'seed': seed,
        'mode': mode,
        'n_neighbors': n_neighbors,
        'anomalies': anomalies.tolist(),
    }
    
    if plot:
        plot_results(results)
    
    return results

def kmeans_clustering(data, mode, seed=None, plot=True):
    k = estimate_k(data)
    
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    results = {
        'method': 'KMeans Clustering',
        'seed': seed,
        'mode': mode,
        'clusters': labels.tolist(),
        'param_estimation': {'k': k}
    }
    
    if plot:
        plot_results(results)
    
    return results

def louvain_method(data, mode, seed=None, plot=True):
    # Ensure data is in graph form for 'rn' mode; for other modes, additional preprocessing might be required
    
    if mode != 'rn':
        print("Louvain method requires data in recurrence network format.")
        return
    
    partition = community_louvain.best_partition(data, random_state=seed)
    
    results = {
        'method': 'Louvain Method',
        'seed': seed,
        'mode': mode,
        'communities': partition,
        'param_estimation': 'default'
    }
    
    if plot:
        plot_results(results)
    
    return results

def girvan_newman_community(data, mode, seed=None, plot=True):
    if mode not in ['rn', 'vg']:
        raise ValueError("Girvan-Newman method requires network data.")
    
    communities_generator = nx.community.girvan_newman(data)
    
    # maybe decide on a criterion to stop division, for now, take the first division
    communities = next(communities_generator)

    results = {
        'method': 'Girvan-Newman',
        'seed': seed,
        'mode': mode,
        'communities': list(map(list, communities))
    }

    if plot:
        plot_results(results)
    
    return results

def spectral_clustering(data, mode, seed=None, n_clusters=None, plot=True):
    if n_clusters is None:
        n_clusters = estimate_k(data)  # Reuse KMeans k estimation function
    
    spectral_model = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='nearest_neighbors')
    labels = spectral_model.fit_predict(data)

    results = {
        'method': 'Spectral Clustering',
        'seed': seed,
        'mode': mode,
        'clusters': labels.tolist(),
        'param_estimation': {'n_clusters': n_clusters}
    }

    if plot:
        plot_results(results)
    
    return results

def one_class_svm(data, mode, seed=None, nu=0.05, gamma='scale', plot=True):
    oc_svm = OneClassSVM(nu=nu, gamma=gamma, random_state=seed)
    anomalies = oc_svm.fit_predict(data)

    results = {
        'method': 'One-Class SVM',
        'seed': seed,
        'mode': mode,
        'anomalies': anomalies.tolist(),
        'param_estimation': {'nu': nu, 'gamma': gamma}
    }

    if plot:
        plot_results(results)
    
    return results

def dbscan_clustering(data, mode, seed=None, eps=None, min_samples=None, plot=True):
    if eps is None or min_samples is None:
        best_params = iterative_dbscan_param_estimation(data)
        eps = best_params['eps']
        min_samples = best_params['min_samples']
    
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan_model.fit_predict(data)

    results = {
        'method': 'DBSCAN',
        'seed': seed,
        'mode': mode,
        'clusters': labels.tolist(),
        'param_estimation': {'eps': eps, 'min_samples': min_samples}
    }

    if plot:
        plot_results(results)
    
    return results

def infomap_community(data, mode, plot=True):
    if mode not in ['rn', 'vg']:
        raise ValueError("Infomap requires network data.")
    
    im = infomap.Infomap()
    
    # Assuming 'data' is a NetworkX graph, add edges to Infomap
    for edge in data.edges():
        im.addLink(*edge)
    
    im.run()
    
    communities = {}
    for node in im.tree:
        if node.isLeaf():
            communities[node.node_id] = node.module_id
    
    results = {
        'method': 'Infomap',
        'mode': mode,
        'communities': communities
    }
    
    if plot:
        plot_results(results)
    
    return results

def autoencoder_anomaly_detection(data, mode, epochs=50, plot=True):
    # Preprocess data: scale features between 0 and 1
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Define a simple autoencoder architecture
    n_features = data.shape[1]
    autoencoder = Sequential([
        Input(shape=(n_features,)),
        Dense(n_features * 2, activation='relu'),
        Dense(n_features, activation='relu'),
        Dense(n_features // 2, activation='relu'),
        Dense(n_features, activation='sigmoid')
    ])
    
    autoencoder.compile(optimizer=Adam(), loss='mse')
    
    # Train the autoencoder
    autoencoder.fit(data_scaled, data_scaled, epochs=epochs, verbose=0)
    
    # Use the autoencoder for anomaly detection
    reconstructions = autoencoder.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    mse_threshold = np.percentile(mse, 95)  # Consider top 5% MSE as anomalies
    anomalies = mse > mse_threshold
    
    results = {
        'method': 'Autoencoder',
        'mode': mode,
        'anomalies': anomalies.tolist(),
        'param_estimation': {'epochs': epochs, 'mse_threshold': mse_threshold}
    }

    if plot:
        plot_results(results)
    
    return results

def dynamic_modularity_optimization(network_snapshots):
    """
    Apply Dynamic Modularity Optimization across network snapshots.
    """
    communities_list = [detect_communities_louvain(snapshot) for snapshot in network_snapshots]
    

    return aligned_communities

def dynamic_modularity_optimization(network_snapshots, plot=True):
    """
    Apply Dynamic Modularity Optimization across network snapshots.
    """
    
    all_communities = [detect_communities_louvain(snapshot) for snapshot in network_snapshots]
    
    # Alignment of communities across snapshots would go here
    aligned_communities = align_communities(all_communities)
    
    results = {
        'method': 'Dynamic Modularity Optimization',
        'mode': 'temporal networks',
        'communities': all_communities,
        'alignment': aligned_communities,  # After implementing alignment
    }

    if plot:
        plot_results(results)  # would need to adapt this for temporal data
    
    return results

def dynamic_community_detection(network_snapshots):
    """
    Apply a dynamic community detection algorithm on a list of networkx graphs representing network snapshots.
    """
    temporal_network = TemporalClustering(graphs=network_snapshots)
    
    # Example with a static method over snapshots, for dynamic methods adjust accordingly
    for snapshot in network_snapshots:
        communities = algorithms.louvain(snapshot)
        temporal_network.add_clustering(communities)
    
    # Evaluate the quality of the partition
    scores = [evaluation.newman_girvan_modularity(snapshot, communities).score for snapshot, communities in zip(network_snapshots, temporal_network.clusterings)]
    
    return temporal_network, scores