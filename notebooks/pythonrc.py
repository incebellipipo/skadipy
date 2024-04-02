# This is for the notebook to reload external python modules

import sys


sys.path.insert(0, "../src")

from save_data import *
import matplotlib.pyplot as plt
import typing

import skadipy.allocator.reference_filters
import skadipy.actuator
import skadipy.allocator
import skadipy.toolbox
import skadipy.safety

# Importing the necessary modules

# Setting the plot style
plt.rcParams['text.usetex'] = True

# Creating the vessel
tunnel = skadipy.actuator.Fixed(
    position=skadipy.toolbox.Point([0.3875, 0.0, 0.0]),
    orientation=skadipy.toolbox.Quaternion(
        axis=(0.0, 0.0, 1.0), radians=np.pi / 2.0),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "limits": [-1.0, 1.0]
    }
)
voithschneider_port = skadipy.actuator.Azimuth(
    position=skadipy.toolbox.Point([-0.4574, -0.055, 0.0]),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "reference_angle":  np.pi / 4.0,
        "limits": [0.0, 1.0]
    }
)
voithschneider_starboard = skadipy.actuator.Azimuth(
    position=skadipy.toolbox.Point([-0.4547, 0.055, 0.0]),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "reference_angle": - np.pi / 4.0,
        "limits": [0.0, 1.0]
    }
)

ma_bow_port = skadipy.actuator.Fixed(
    position=skadipy.toolbox.Point([0.3, -0.1, 0.0]),
    orientation=skadipy.toolbox.Quaternion(
        axis=(0.0, 0.0, 1.0), angle=(3*np.pi / 4.0)),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "limits": [0.0, 1.0]
    }
)
ma_bow_starboard = skadipy.actuator.Fixed(
    position=skadipy.toolbox.Point([0.3, 0.1, 0.0]),
    orientation=skadipy.toolbox.Quaternion(
        axis=(0.0, 0.0, 1.0), angle=(-3*np.pi / 4.0)),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "limits": [0.0, 1.0]
    }
)
ma_aft_port = skadipy.actuator.Fixed(
    position=skadipy.toolbox.Point([-0.3, -0.1, 0.0]),
    orientation=skadipy.toolbox.Quaternion(
        axis=(0.0, 0.0, 1.0), angle=(np.pi / 4.0)),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "limits": [0.0, 1.0]
    }
)
ma_aft_starboard = skadipy.actuator.Fixed(
    position=skadipy.toolbox.Point([-0.3, 0.1, 0.0]),
    orientation=skadipy.toolbox.Quaternion(
        axis=(0.0, 0.0, 1.0), angle=(-np.pi / 4.0)),
    extra_attributes={
        "rate_limit": 0.1,
        "saturation_limit": 1.0,
        "limits": [0.0, 1.0]
    }
)


def generate_spiral_dataset(num_points: int, num_turns: float, k: float = 1.0) -> np.ndarray:
    """
    Generate a dataset of points on a spiral.

    :param num_points: The number of points to generate.
    :param num_turns: The number of turns to make.
    :param k: The scaling factor for the spiral.
    """
    angles = np.linspace(0, 2 * np.pi * num_turns, num_points)
    radii = np.linspace(0, 1, num_points)
    return np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii)) * k


def gen_clipped_sin(n: int, amplitude: float, period: float, phase: float, offset: float, clip_n: float = -1.0, clip_p: float = 1.0):
    """
    Generate a clipped sine wave.

    :param n: The number of points to generate.
    :param amplitude: The amplitude of the sine wave.
    :param period: The period of the sine wave.
    :param phase: The phase of the sine wave.
    :param offset: The offset of the sine wave.
    :param clip_n: The negative clipping value.
    :param clip_p: The positive clipping value.

    :return: The clipped sine wave.
    """
    return np.clip(amplitude * np.sin(np.linspace(0, 2 * np.pi * period, n) + phase) + offset, clip_n, clip_p)


def run_tests(tau_cmd: np.ndarray = None, d_tau_cmd: np.ndarray = None, allocators: typing.List[
    skadipy.allocator.AllocatorBase] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the tests for the given allocator and input forces.

    :param tau_cmd: The input forces to the system. This is a 2D array with shape (N, 6), where N is the number of samples.
    :param d_tau_cmd: The derivative of the input forces to the system. This is a 2D array with shape (N, 6), where N is the number of samples.
    :param allocators: The list of allocator to test.
    :return: A tuple with the following elements:
        - xi_hist: A 3D array with shape (len(allocator), N, n), where n is the number of inputs to the allocator.
        - theta_hist: A 3D array with shape (len(allocator), N, q), where q is the number of outputs from the allocator.
        - tau_hist: A 3D array with shape (len(allocator), N, 6), where 6 is the number of force components.
    """
    # get the number of samples
    N = tau_cmd.shape[0]

    # Prepare the allocator
    for i in allocators:
        i.compute_configuration_matrix()

    # Get the first allocator's B matrix and compute the number of inputs and outputs
    dof_indices = [i.value for i in allocators[0].force_torque_components]
    B = allocators[0]._b_matrix[dof_indices, :]
    n = B.shape[1]
    q = B.shape[1] - B.shape[0]
    del B, dof_indices

    # Prepare the history arrays
    xi_hist = np.zeros((len(allocators), N, n))
    theta_hist = np.zeros((len(allocators), N, q))
    tau_hist = np.zeros((len(allocators), N, 6))

    # Run the tests
    for i, allocator in enumerate(allocators):
        for j in range(N):

            kwargs = {}
            kwargs['tau'] = np.reshape(tau_cmd[j], (6, 1))
            if d_tau_cmd is not None:
                kwargs['d_tau'] = np.reshape(d_tau_cmd[j], (6, 1))

            xi, theta = allocator.allocate(**kwargs)

            xi_hist[i, j, :] = xi.flatten()
            if theta is not None:
                theta_hist[i, j, :] = theta.flatten()
                tau_hist[i, j, :] = allocator.allocated.flatten()

    # Return the results
    return (xi_hist, theta_hist, tau_hist)

def plot_histories(tau_cmd, tau_alloc, indices=[0, 1, 5]):
    labels = ['X', 'Y', 'Z', 'K', 'M', 'N']
    # plt.figure(figsize=(20, 20))
    for i in range(len(indices)):
        plt.subplot(3, 1, i + 1)
        plt.plot(tau_cmd[:, indices[i]], label='Input $' + labels[indices[i]] + '$', linewidth=1, linestyle='-', color='black')

        for j in range(len(tau_alloc)):
            plt.plot(tau_alloc[j][:, indices[i]], label='Output $' + labels[indices[i]] + '$', linewidth=1, linestyle='-')

        plt.grid(True)
        plt.legend()

    plt.show()

def plot_2d_allocation(tau_cmd: np.ndarray, allocators: typing.List[skadipy.allocator.AllocatorBase], tau_hist: np.ndarray):
    plt.scatter(tau_cmd[:, 0], tau_cmd[:, 1], s=5, label="Input forces", color='black')

    colors = ['b','r','c']
    for allocator, F,color in zip(allocators, tau_hist, colors):
        plt.scatter(F[:, 0], F[:, 1], s=5, color=color)
        for i in range(len(tau_cmd)):
            plt.plot([tau_cmd[i, 0], F[i, 0]], [tau_cmd[i, 1], F[i, 1]], "--", color=color, lw=0.5)