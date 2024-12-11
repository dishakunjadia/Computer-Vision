import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    dists = np.linalg.norm(features1[:, np.newaxis] - features2, axis=2)

    #raise NotImplementedError('`match_features` function in ' +
        #'`student_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in
    section 7.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    dists = compute_feature_distances(features1, features2)
    sorted_indices = np.argsort(dists, axis=1)
    nearest_neighbor_idx = sorted_indices[:, 0]  # Index of the nearest neighbor in features2
    second_nearest_idx = sorted_indices[:, 1]    # Index of the second nearest neighbor in features2

    # Step 3: Compute the ratio of the nearest distance to the second nearest distance
    nearest_dist = dists[np.arange(dists.shape[0]), nearest_neighbor_idx]
    second_nearest_dist = dists[np.arange(dists.shape[0]), second_nearest_idx]
    ratio = nearest_dist / second_nearest_dist

    # Step 4: Apply the ratio test (Lowe's ratio test)
    ratio_threshold = 0.78
    valid_matches_mask = ratio < ratio_threshold

    # Step 5: Gather valid matches and their corresponding confidences
    matches = np.column_stack((np.where(valid_matches_mask)[0], nearest_neighbor_idx[valid_matches_mask]))
    confidences = nearest_dist[valid_matches_mask]

    # Step 6: Sort matches by confidence in descending order
    sorted_confidence_indices = np.argsort(confidences)[::-1]
    matches = matches[sorted_confidence_indices]
    confidences = confidences[sorted_confidence_indices]


    
    #raise NotImplementedError('`match_features` function in ' +
#'`student_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
