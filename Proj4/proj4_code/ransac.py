import numpy as np
import math
from least_squares_fundamental_matrix import solve_F
import two_view_data
import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    num_samples = math.log(1-prob_success)/math.log(1-(ind_prob_correct)**sample_size)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return int(num_samples)


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    
    if x_0s.shape[1] == 2:
        x_0s = np.hstack((x_0s, np.ones((x_0s.shape[0], 1))))
    if x_1s.shape[1] == 2:
        x_1s = np.hstack((x_1s, np.ones((x_1s.shape[0], 1))))
    error = fundamental_matrix.signed_point_line_errors(x_0s, F, x_1s)
    error = np.abs(np.array(error))

    error_left = error[0:error.shape[0]:2]
    error_right = error[1:error.shape[0]:2]
    total_error = (error_left + error_right) / 2

    inliers = np.argwhere(threshold >= total_error).flatten()
    
  
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return inliers


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

  
    ransac_iterations = calculate_num_ransac_iterations(0.99, 9, 0.9)
    best_F = None
    max_inliers = 0

    for _ in range(ransac_iterations):
        sample_indices = np.random.choice(x_0s.shape[0], 9, replace=False)
        sample_left = np.array(x_0s[sample_indices])
        sample_right = np.array(x_1s[sample_indices])
        
        candidate_F = solve_F(sample_left, sample_right)
        
        current_inliers = find_inliers(x_0s, candidate_F, x_1s, threshold=1)
        num_inliers = len(current_inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = candidate_F
            inliers_x_0 = x_0s[current_inliers]
            inliers_x_1 = x_1s[current_inliers]
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return best_F, inliers_x_0, inliers_x_1


def test_with_epipolar_lines():
    """Unit test you will create for your RANSAC implementation.

    It should take no arguments and it does not need to return anything,
    but it **must** display the images when run.

    Use the code in the jupyter notebook as an example for how to open the
    image files and perform the necessary operations on them in our workflow.
    Remember the steps are Harris, SIFT, match features, RANSAC fundamental matrix.

    Display the proposed correspondences, the true inlier correspondences
    found by RANSAC, and most importantly the epipolar lines in both of your images.
    It should be clear that the epipolar lines intersect where the second image
    was taken, and the true point correspondences should indeed be good matches.

    """
    from feature_matching.SIFTNet import get_siftnet_features
    from feature_matching.utils import load_image, PIL_resize, rgb2gray
    from feature_matching.HarrisNet import get_interest_points
    from feature_matching.utils import show_interest_points
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from feature_matching.utils import load_image
    from feature_matching.utils import load_image


    from utils import draw_epipolar_lines


    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    image1 = load_image('../data/ransac_test_0.jpg')
    image_right = load_image('../data/ransac_test_1.jpg')


    scaling = 0.5
    image_left_resized = PIL_resize(
        image1, (int(image1.shape[1] * scaling), int(image1.shape[0] * scaling)))
    image_right_resized = PIL_resize(
        image2, (int(image2.shape[1] * scaling), int(image2.shape[0] * scaling)))

    image_left_bw = rgb2gray(image_left_resized)
    image_right_bw = rgb2gray(image_right_resized)

    tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    to_tensor = transforms.ToTensor()
    tensor_left = to_tensor(image_left_bw).unsqueeze(0)
    tensor_right = to_tensor(image_right_bw).unsqueeze(0)


    corners_x_left, corners_y_left, _ = get_interest_points(tensor_left.float())
    corners_x_right, corners_y_right, _ = get_interest_points(tensor_right.float())

    corners_x_left, corners_x_right = corners_x_left.detach().numpy(), corners_x_right.detach().numpy()
    corners_y_left, corners_y_right = corners_y_left.detach().numpy(), corners_y_right.detach().numpy()
    print(f'{len(corners_x_left)} corners in the left image, {len(corners_x_right)} corners in the right image')

    image_left_descriptors = get_siftnet_features(tensor_left, corners_x_left, corners_y_left)
    image_right_descriptors = get_siftnet_features(tensor_right, corners_x_right, corners_y_right)

    from feature_matching.student_feature_matching import match_features
    matched_indices, match_confidences = match_features(
    image_left_descriptors, image_right_descriptors, corners_x_left, corners_y_left, corners_x_right, corners_y_right)
    print(f'{len(matched_indices)} matches found from {len(corners_x_left)} corners')
    from feature_matching.utils import show_correspondence_circles, show_correspondence_lines
    num_matches_to_visualize = 100
    matches_visualization = show_correspondence_lines(
    image_left_resized, image_right_resized,
    corners_x_left[matched_indices[:num_matches_to_visualize, 0]], corners_y_left[matched_indices[:num_matches_to_visualize, 0]],
    corners_x_right[matched_indices[:num_matches_to_visualize, 1]], corners_y_right[matched_indices[:num_matches_to_visualize, 1]])
    

    plt.figure()

    plt.title('Proposed Matches')
    plt.imshow(matches_visualization)

    left_points = np.zeros((len(matched_indices), 2))
    right_points = np.zeros((len(matched_indices), 2))
    left_points[:, 0] = corners_x_left[matched_indices[:, 0]]
    left_points[:, 1] = corners_y_left[matched_indices[:, 0]]
    right_points[:, 0] = corners_x_right[matched_indices[:, 1]]
    right_points[:, 1] = corners_y_right[matched_indices[:, 1]]
    
    F_matrix, inliers_left_points, inliers_right_points = ransac_fundamental_matrix(left_points, right_points)
    print("Estimated Fundamental Matrix:\n", F_matrix)

    true_matches_visualization = show_correspondence_lines(
    image_left_resized, image_right_resized,
    inliers_left_points[:num_matches_to_visualize, 0], inliers_left_points[:num_matches_to_visualize, 1],
    inliers_right_points[:num_matches_to_visualize, 0], inliers_right_points[:num_matches_to_visualize, 1])

    plt.figure()
    plt.title('True Matches')
    plt.imshow(true_matches_visualization)
    draw_epipolar_lines(F_matrix, image_left_resized, image_right_resized, inliers_left_points, inliers_right_points)


    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
