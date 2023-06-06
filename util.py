import numpy as np
from scipy.spatial.distance import cdist, euclidean


def load_categories(file_path):
    categories = []
    with open(file_path) as file:
        for category in file:
            categories.append(category.strip())
    return categories


def find_maximal_subset(vector_dict):
    words = list(vector_dict.keys())
    vectors = list(vector_dict.values())
    selected_subset = [words[0]]
    distances = cdist([vectors[0]], vectors).min(axis=0)

    for _ in range(1, 25):
        max_distance_idx = np.argmax(distances)
        selected_subset.append(words[max_distance_idx])
        new_distances = cdist([vectors[max_distance_idx]], vectors).min(axis=0)
        distances = np.minimum(distances, new_distances)

    return selected_subset
