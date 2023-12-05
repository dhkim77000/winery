import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import faiss
import pandas as pd
import pdb
import gc
from tqdm import tqdm
import umap.umap_ as umap
from collections import Counter
from sklearn.metrics import silhouette_score

def get_nns(inter_wine_ids: pd.DataFrame,
            item_data: pd.DataFrame,
            index: faiss.IndexIDMap2,
            total_k: int = 15000):

    wine_ids = inter_wine_ids
    X = np.array(item_data.loc[wine_ids, 'vectors'].tolist(), dtype=np.float32)
    vectors = item_data.loc[wine_ids, 'vectors']
    
    
    umap_model =  umap.UMAP(init = 'random', n_components=256)
    X_reduced = umap_model.fit_transform(X)

    # Determine the optimal number of clusters using the Silhouette Score
    max_k = len(vectors) if len(vectors) < 12 else 12
    best_silhouette_score = -1
    optimal_k = 1

    
    for k in tqdm(range(2, max_k + 1)):
        nn = NearestNeighbors(n_neighbors=k).fit(X_reduced)
        distances, indices = nn.kneighbors(X_reduced)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        num_elements = len(distances)
        lower_10_percent_idx = int(num_elements * 0.1)
        upper_10_percent_idx = int(num_elements * 0.9)
        
        middle_data = distances[lower_10_percent_idx:upper_10_percent_idx]

        min_value = middle_data.min()
        max_value = middle_data.max()

        bin_edges = np.linspace(min_value, max_value, 5 + 1)

        for eps in bin_edges:
            clustering = DBSCAN(eps=eps, min_samples=k)
            #kmeans = KMeans(n_clusters=k, random_state=42)
            clustering.fit(X_reduced)
            labels = clustering.labels_

            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(X_reduced, labels)
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    optimal_k = k
                    optimal_eps = eps
            else: continue

    # Step 4: Apply K-Means with the optimal K value
    clustering = DBSCAN(eps=optimal_eps, min_samples=optimal_k)
    #kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = clustering.fit_predict(X_reduced)
    print(len(set(clusters)))
    cluster_k = Counter(clusters)

    num_cluster = len(set(clusters))
    k = total_k // num_cluster

    for cluster, count in cluster_k.items():
        cluster_k[cluster] = int(total_k / len(clusters)) * count

    mean_vectors = {}
    for vector, cluster in zip(vectors, clusters):
        if cluster not in vector:
            mean_vectors[cluster] = {'count': 1, 'mean': vector}
        else:
            mean_vectors[cluster]['count'] += 1
            mean_vectors[cluster]['mean'] += (vector - mean_vectors[cluster]['mean']) / mean_vectors[cluster]['count']

    result = []
    for cluster, to_search in tqdm(mean_vectors.items()):
        # Faiss expects the query vectors to be normalized
        to_search = to_search['mean']
        to_search = np.expand_dims(to_search, axis=0)
        to_search = np.ascontiguousarray(to_search.astype(np.float32))

        distances, searched_wine_ids = index.search(to_search, k=cluster_k[cluster])

        for dis, id in zip(distances[0], searched_wine_ids[0]):
            result.append((id, dis))


    del mean_vectors, X, X_reduced, clusters, clustering, umap_model, distances, searched_wine_ids
    gc.collect()

    result.sort(key=lambda x: x[1])



    return result

def get_nns_elbow(user : str,
            inter_per_user : pd.DataFrame,
            item_data: pd.DataFrame,
            index:faiss.IndexIDMap2,
            total_k: int = 15000):
    
    wine_ids = inter_per_user[user]


    vectors = item_data.loc[wine_ids, 'vectors']

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    X = vectorizer.fit_transform(vectors).toarray()

    # Step 2: Normalize the vectors
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    max_k = len(vectors) if len(vectors) < 12 else 12
    k_range = range(1, max_k)  # Test different K values from 1 to 10
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_normalized)
        inertia.append(kmeans.inertia_)

    # Determine the optimal K value (Elbow point)
    optimal_k = np.argmin(np.diff(inertia)) + 1

    # Step 4: Apply K-Means with the optimal K value
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_normalized)
    cluster_k = Counter(clusters)

    num_cluster = len(mean_vectors)
    k = total_k // num_cluster
    
    for cluster, count in cluster_k.items():
        cluster_k[cluster] = int(total_k / len(clusters)) * count

    mean_vectors = {}
    for vector, cluster in zip(vectors, clusters):
        if cluster not in vector:
            mean_vectors[cluster] = {'count': 1, 'mean': vector}
        else:
            mean_vectors[cluster]['count'] += 1
            mean_vectors[cluster]['mean'] += (vector - mean_vectors[cluster]['mean']) / mean_vectors[cluster]['count']

    

    

    result = []
    for cluster, to_search in tqdm(mean_vectors.items()):
        # Faiss expects the query vectors to be normalized
        to_search  = to_search['mean']
        to_search = np.expand_dims(to_search, axis=0)
        to_search = np.ascontiguousarray(to_search.astype(np.float32))

        distances, searched_wine_ids = index.search(to_search, k= cluster_k[cluster])

        for dis, id in zip(distances[0], searched_wine_ids[0]):
            result.append((id, dis))
    
    result.sort(key = lambda x: x[1])
    return result