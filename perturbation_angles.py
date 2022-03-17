import numpy as np
import math


def generate_perturbation_angles(correlation_length, dz, fiber_length):
    num_points = math.ceil(fiber_length / dz)
    sigma = 1 / np.sqrt(2 * correlation_length)
    buffer = np.sqrt(dz) * sigma * np.random.randn(num_points)

    theta0 = np.random.rand(1) * 2 * np.pi
    buffer[0] = theta0
    thetas = np.cumsum(buffer)
    return thetas
