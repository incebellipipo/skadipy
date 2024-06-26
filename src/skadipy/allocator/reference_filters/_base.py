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

import typing
import numpy as np
import scipy


from ... import toolbox
from ... import safety
from ... import allocator
from ... import actuator

class ReferenceFilterBase(allocator.AllocatorBase):
    def __init__(
        self,
        actuators: typing.List[actuator.ActuatorBase] = (),
        force_torque_components: typing.List[allocator.ForceTorqueComponent] = None,
        gamma: float = 1.0,
        mu: float = 1.0,
        rho: float = 1.0,
        zeta: float = 1.0,
        control_barrier_function = safety.ControlBarrierFunctionType.ABSOLUTE,
        time_step: float = 0.1,
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
            actuators=actuators, force_torque_components=force_torque_components
        )

        # Steepest descent gain
        self._gamma = gamma

        # maneuvering gradient update law gain, choosing mu > gamma
        self._mu = mu

        # Safety gain to avoid unsafe set (saturation limits)
        self._rho = rho

        # Zeta, regularization parameter
        self._zeta = zeta

        # B matrix
        self._b_matrix_weighted_inverse = None

        # Q matrix, nullspace of B
        self._q_matrix = None
        # Theta, to be multiplied with Q, where
        # $$
        # $Q\coloneqq \mathcal{N}(B)$ \\
        # Q in \mathbb{R}^{p \times q}
        # \theta \in \mathbb{R}^q
        # $$
        self._theta = None

        # Kappa method
        self._control_barrier_function = control_barrier_function

        self._xi_virtual = None

        # Time step
        self._t_s = time_step

    def _kappa(self, xi, xi_error, d_xi_particular, upsilon) -> np.ndarray:
        r"""
        Compute the safe control input, :math:`\dot{\xi}`.

        .. math::
            :nowrap:

            \begin{align*}
            & \kappa := - \bar{U} \frac{\tilde{\xi}}{||\tilde{\xi}|| + \\
                \varepsilon} + \dot{\xi}_p + Q \upsilon \\
            & 0 < \bar{U} <= ||\dot{\xi}||
            \end{align*}


        :param xi_error: Error in the allocated forces
        :param d_xi_particular: Derivative of the particular solution
        :param upsilon:
        :return: Change in allocated forces
        """
        kappa = np.zeros_like(xi_error)
        # xi_u = np.zeros_like(xi_error)

        # d_xi = d_xi_particular
        d_xi = np.array(d_xi_particular, copy=True)
        i = 0
        for a in self._actuators:
            # Get contribution matrix
            _, cols = np.shape(a.B)


            ###
            # cvompute the Control Lyapunov Function
            ##

            # get the rate limit
            # :math:`0 < \bar{U} <= ||\dot{\xi}||`.
            #
            rate_limit = a.extra_attributes.get("rate_limit", np.inf)

            xi_i_error = xi_error[i: i + cols]

            xi_i_error_norm = np.linalg.norm(xi_i_error)

            kappa[i:i+cols] += - rate_limit * (xi_i_error / (xi_i_error_norm + self._zeta))

            kappa[i:i+cols] += d_xi[i: i + cols]

            kappa[i:i+cols] += self._q_matrix[i: i + cols, :] @ upsilon

            ##
            # compute the Control Barrier Function
            ##

            if self._control_barrier_function == safety.ControlBarrierFunctionType.SUMSQUARE:
                kappa[i:i+cols] = safety.sumsquare(xi[i: i + cols], kappa[i: i + cols], self._rho, a)

            elif self._control_barrier_function == safety.ControlBarrierFunctionType.ABSOLUTE:
                kappa[i:i+cols] = safety.absolute(xi[i: i + cols], kappa[i: i + cols], self._rho, a)

            elif self._control_barrier_function == safety.ControlBarrierFunctionType.QUADRATIC:
                kappa[i:i+cols] = safety.quadratic(xi[i: i + cols], kappa[i: i + cols], self._rho, a)

            # Increment it as much as the columns in the capability matrix
            i += cols

        # kappa := xi_dot
        return kappa

    def compute_configuration_matrix(self) -> None:
        super().compute_configuration_matrix()

        # Indices for degrees of freedom
        dof_indices = [i.value for i in self.force_torque_components]

        # Prepare the required variables
        b_matrix = self._b_matrix[dof_indices, :]
        w_matrix = self._w_matrix
        self._b_matrix_weighted_inverse = toolbox.weighted_pseudo_inverse(
            matrix=b_matrix, weights=w_matrix)

        # Nullspace of B, $Q \in \mathcal{N}(B)$
        self._q_matrix = scipy.linalg.null_space(b_matrix)