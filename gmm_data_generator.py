import numpy as np
import itertools


def load_db(num_data: int = 512, standard_deviation: float = 0.01, max_point=5, min_point=-5, step=2):
    """
    :param num_data:
    :param standard_deviation:
    :param max_point:
    :param min_point:
    :param step:
    :return:
    """
    thetas = np.linspace(0, 1.75*np.pi, 8)
    xs, ys = 1 * np.sin(thetas), 1 * np.cos(thetas)
    # mu_vector = np.array([np.array([i, j]) for i, j in itertools.product(np.arange(min_point, max_point, step),
    #                                                                      np.arange(min_point, max_point, step))],
    #                      dtype=np.float32)
    # variance = (standard_deviation ** 2) * np.ones(mu_vector.shape[0])
    # return generate_2d_gmm(num_data, mu_vector, variance)
    n_mix = 8  # mu_vector.shape[0]
    num_data_per_mixture = num_data // n_mix
    i_matrix = np.eye(2)
    variance = (standard_deviation ** 2) * np.ones(2)
    return np.concatenate(
        [np.random.multivariate_normal([xi,yi], i_matrix * variance, num_data_per_mixture) for xi,yi in
         zip(xs.ravel(), ys.ravel())]).astype(np.float32)


def load_mixture_of_two_gaussian(num_data: int = 10000, standard_deviation: float = 0.05, point_a=5, point_b=-5):
    mu_vector = np.zeros([2, 2])
    mu_vector[0, :] = point_a
    mu_vector[1, :] = point_b
    variance = (standard_deviation ** 2) * np.ones(mu_vector.shape[0])
    return generate_2d_gmm(num_data, mu_vector, variance)


def generate_2d_gmm(num_data, mu_vector: np.array, variance_vector):
    """
    :param num_data:
    :param mu_vector:
    :param variance_vector:
    :return:
    """
    assert len(mu_vector.shape) == 2, 'Mu vector shape must be of size 2'
    assert mu_vector.shape[1] == 2, 'Mu vector shape[1] must be equal to 2'
    assert len(variance_vector.shape) == 1, 'Variance vector shape must be of size 1'
    assert variance_vector.shape[0] == mu_vector.shape[
        0], 'mu vector and variance vector must agree on the number of mixtures'
    n_mix = 8  #mu_vector.shape[0]
    num_data_per_mixture = num_data // n_mix
    i_matrix = np.eye(2)
    return np.concatenate(
        [np.random.multivariate_normal(mu_vector[i, :], i_matrix * variance_vector[i], num_data_per_mixture) for i in
         range(n_mix)]).astype(np.float32)