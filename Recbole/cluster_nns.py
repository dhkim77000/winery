import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import faiss
import pandas as pd
from tqdm import tqdm
from collections import Counter
def get_nns(user : str,
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