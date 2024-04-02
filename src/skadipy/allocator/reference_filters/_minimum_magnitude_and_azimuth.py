# This file is part of skadipy.
#
# skadi is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# skadi is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# skadipy. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) 2024 Emir Cem Gezer, NTNU

import numpy as np
import scipy
import typing

from ._base import ReferenceFilterBase
from ...safety import *
from ... import allocator
from ... import actuator
from ... import toolbox

class MinimumMagnitudeAndAzimuth(ReferenceFilterBase):

    def __init__(
            self,
            actuators: typing.List[actuator.ActuatorBase] = (),
            force_torque_components: typing.List[allocator.ForceTorqueComponent] = None,
            gamma: float = 1e-2,
            mu: float = 1e-2,
            rho: float = 1.0,
            zeta: float = 0.1,
            control_barrier_function: ControlBarrierFunctionType = ControlBarrierFunctionType.ABSOLUTE,
            derivative: toolbox.derivative.DerivativeBase = toolbox.derivative.ExponentialSmoothing(r=0.1),
            time_step: float = 0.001,
            lambda_: float = 0.9
    ) -> None:
        r"""
        Minimum magnitude reference filter allocator.

        :param actuators: List of actuators
        :param force_torque_components: List of force-torque components
        :param gamma: Steepest descent gain
        :param mu: maneuvering gradient update law gain, choosing mu > gamma
        :param rho: Safety gain to avoid unsafe set (saturation limits)
        """
        super().__init__(
            actuators=actuators,
            force_torque_components=force_torque_components,
            gamma=gamma,
            mu=mu,
            rho=rho,
            zeta=zeta,
            control_barrier_function=control_barrier_function,
            time_step=time_step
        )

        self._derivative_solver = derivative

        self._lambda = lambda_

        self._xi = None

    def allocate(self, tau: np.ndarray, d_tau: np.ndarray = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        """

        :param tau:
        :return:
        """

        # Indices for degrees of freedom
        dof_indices = [i.value for i in self.force_torque_components]

        # Prepare the required variables
        b_matrix = self._b_matrix[dof_indices, :]
        w_matrix = self._w_matrix
        b_weighted_inverse = toolbox.weighted_pseudo_inverse(
            matrix=b_matrix, weights=w_matrix)

        # Nullspace of B, $Q \in \mathcal{N}(B)$
        self._q_matrix = scipy.linalg.null_space(b_matrix)

        if self._theta is None:
            # Initialization of variables
            n, p = np.shape(b_matrix)
            # Theta is the parameter vector for the maneuvering gradient
            q = p - n
            self._theta = np.zeros((q, 1), dtype=np.float32)

        # Initialize the virtual force/torque
        if self._xi is None:
            self._xi = np.zeros(
                (b_matrix.shape[1], 1), dtype=np.float32)

        # Compute $\dot{xi}_p and use exponential smoothing
        # Check if it has been provided to the function
        if type(d_tau) == type(None):
            d_tau = self._derivative_solver(tau) / self._t_s

        # Compute the particular solution
        xi_p = b_weighted_inverse @ tau[dof_indices, :]

        # Compute the derivative of particular solution using derivative of
        # requested force
        d_xi_p = b_weighted_inverse @ d_tau[dof_indices, :]

        # Get the desired allocated forces on the manifold
        xi_d = xi_p + self._q_matrix @ self._theta

        # Compute the error
        xi_error = self._xi - xi_d

        # Compute the change in theta
        upsilon = - self._gamma * (self.__j_theta(xi_desired=xi_d)).T
        theta_dot = upsilon - self._mu * (self.__v_theta(xi_error).T)

        # Advance theta as much as the change in theta
        self._theta = self._theta + theta_dot * self._t_s

        # Compute the change control input, xi
        kappa = self._kappa(self._xi, xi_error, d_xi_p, upsilon)

        self._xi = self._xi + kappa * self._t_s

        self._command(self._xi)

        return self._xi, self._theta

    def __j_theta(self, xi_desired: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of the cost function with respect to theta.

        :param xi_desired: Desired force/torque on the manifold
        """

        j_theta = np.zeros_like(self._theta).T

        i = 0
        for actuator in self._actuators:
            _, cols = actuator.B.shape

            # Get the desired force/torque for the actuator
            xi_di = xi_desired[i:i + cols]

            xi_ref = self._xi[i:i + cols]

            # Get the weight of the actuator. If it is not specified, use 1.0
            w = actuator.extra_attributes.get('w', 1.0)

            xi_desired_norm = np.linalg.norm(xi_di)
            xi_ref_norm = np.linalg.norm(xi_ref)

            if xi_desired_norm != 0.0:
                if cols == 1:
                    # This is for non-azimuthal actuators
                    sigma_ref = actuator.extra_attributes.get(
                        'reference_direction', None)
                    if sigma_ref == None:
                        sigma_ref = xi_ref / (np.linalg.norm(xi_ref) + toolbox.constants.EPSILON)
                    else:
                        sigma_ref = np.array([sigma_ref])

                    j_theta += \
                        w * (
                                xi_di.T / xi_desired_norm
                                - self._lambda * sigma_ref.T
                            ) @ self._q_matrix[i: i + cols]

                if cols == 2:
                    # This is for azimuthal actuators
                    a_ref = actuator.extra_attributes.get(
                        'reference_angle', None)
                    if a_ref == None:
                        a_ref = xi_ref / (np.linalg.norm(xi_ref) + toolbox.constants.EPSILON)
                    else:
                        a_ref = np.array([np.cos(a_ref), np.sin(a_ref)])

                    j_theta += \
                        w * (xi_di.T / np.linalg.norm(xi_di) -
                             self._lambda * a_ref.T) @ self._q_matrix[i: i + cols]

            i += cols

        return j_theta

    def __v_theta(self, xi_error) -> np.ndarray:
        r"""
        Compute the gradient of the lyapunov control function with respecto to
        theta.

        :param xi_error: Error in the allocated forces
        """
        v_theta = np.zeros_like(self._theta).T

        i = 0
        for actuator in self._actuators:
            _, cols = actuator.B.shape

            v_theta -= (
                actuator.W[0]
                * xi_error[i: i + cols].T @ self._q_matrix[i: i + cols]
            )

            i += cols

        return v_theta
