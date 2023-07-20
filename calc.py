from scipy import spatial
from scipy.stats import ttest_1samp
import numpy as np


def distance(vec1, vec2, distance_measure):
    """
    calculate distance of two word vectors
    :param vec1: vector of first word
    :param vec2: vector of second word
    :param distance_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used similarity/distance
    measure
    :return: result of the distance calculation between the two vectors
    """
    if distance_measure == 'cosine':
        return spatial.distance.cosine(vec1, vec2)
    elif distance_measure == 'manhattan':
        return spatial.distance.cityblock(vec1, vec2)
    elif distance_measure == 'canberra':
        return spatial.distance.canberra(vec1, vec2)
    elif distance_measure == 'euclidian':
        return np.linalg.norm(vec1 - vec2)


def generate_similarities(A, B, distance_measure):
    """
    calculate similarities between two sets of words
    :param A: source set
    :param B: target set
    :param distance_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used similarity/distance
    measure
    :return: list containing all calculated similarities between all the values of source and target set
    """
    distances = []
    for a in A:
        for b in B:
            distances.append(distance(a, b, distance_measure))
    # normalize distance and subtract from 1 to generate similarity
    all_similarities = normalize_and_reverse_distances(distances, distance_measure)
    return all_similarities


def normalize_and_reverse_distances(distances, distance_measure):
    """
    normalize distance calculation results and reverse them so that they represent similarities
    :param distances: list of calculated distances
    :param distance_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used similarity/distance
    measure
    :return: list of similarities between -1 and 1, 1 being most similar, -1 being most dissimilar
    """
    max_distance = max(distances)
    for i in range(len(distances)):
        d = distances[i]
        if distance_measure != 'cosine':
            d_normalized = d / max_distance * 2
        else:
            d_normalized = d
        distances[i] = 1 - d_normalized
    return distances


def t_test(similarities, baseline):
    """
    execute two-sided t-test for set of values and a baseline
    :param similarities: sample observation
    :param baseline: value of the baseline (similarity of first domain set with random sets)
    :return: test statistic (positive or negative), p-value
    """
    return ttest_1samp(similarities, baseline)
