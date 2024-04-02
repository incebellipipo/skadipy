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
import qpsolvers
import scipy
import typing
from ._base import AllocatorBase

class QuadraticProgramming(AllocatorBase):

    def allocate(self, tau: np.ndarray, d_tau: np.ndarray = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        """

        :param tau:
        :return:
        """

        # Get actuator matrices and construct B
        b_matrix_all = np.concatenate([i.B for i in self._actuators], axis=1)

        # Get actuator weights and construct W
        w_matrix = np.diag(sum([i.W for i in self._actuators], []))

        # Get B matrix for the force-torque components
        b_matrix = b_matrix_all[self.force_torque_components, :]

        tau_vector = tau[self.force_torque_components, :]

        # Compute the weighted pseudo-inverse
        f_matrix = self.solve(b_matrix, w_matrix, tau_vector )

        self._command(f_matrix)

        # Return the force matrix
        return f_matrix, None

    def solve(self, b_matrix: np.ndarray, w_matrix: np.ndarray, tau: np.ndarray):

        # Update the problem for the slack variable
        # The slack is applied on the thrust command, not on the individual
        # forces. Interpret number of rows as the number of slack variables
        n_slacks, n_states = np.shape(b_matrix)

        # Append slacks as the new degree of freedom
        constraint_matrix = np.block([b_matrix, np.identity(n_slacks)])

        # Create a weighting matrix for slacks for the cost function
        w_slack = np.eye(n_slacks) * 1e2

        # Update the cost function
        cost_matrix = np.block([
            [w_matrix, np.zeros((n_states, n_slacks))],
            [np.zeros((n_slacks, n_states)), w_slack]
        ])


        P = scipy.sparse.csc.csc_matrix(cost_matrix)
        q = np.zeros(np.shape(cost_matrix)[1])
        A = scipy.sparse.csc.csc_matrix(constraint_matrix)
        b = tau

        # Compute the limits
        lb = []
        ub = []
        for i in self._actuators:
            limits = i.extra_attributes.get('limits', [-np.inf, np.inf])

            lower = min(limits)
            upper = max(limits)
            lb.extend([lower] * len(i.W))
            ub.extend([upper] * len(i.W))

        # Add the slack limits
        lb.extend([-np.inf] * n_slacks)
        ub.extend([np.inf] * n_slacks)

        lb = np.array([lb])
        ub = np.array([ub])

        # Solve the qp in the form of
        #   min     0.5 * x' * P * x + q' * x
        #   s.t.        G * x <= h
        #               A * x = b
        #               lb <= x <= ub
        problem = qpsolvers.Problem(
            P=P,
            q=q,
            G=None,
            h=None,
            A=A,
            b=b,
            lb=lb,
            ub=ub)

        solution = qpsolvers.solve_problem(problem, solver='osqp')

        if solution.found:
            pass
        # allocated_forces = solution.x[0:-n_slacks]
        return solution.x[0: -n_slacks]