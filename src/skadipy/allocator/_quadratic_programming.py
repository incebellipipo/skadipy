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
# Copyright (C) 2026 Emir Cem Gezer, NTNU

import numpy as np
import qpsolvers
import scipy
import typing
from ._base import AllocatorBase

class QuadraticProgramming(AllocatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fd = None

    def _read_fd(self) -> None:
        self._fd = np.zeros(self._b_matrix.shape[1], dtype=np.float64)

        i = 0
        for a in self._actuators:
            total_dofs = a.B.shape[1]
            a_fd = a.extra_attributes.get('desired_force', [0.0] * total_dofs)
            self._fd[i:i + total_dofs] = [a_fd[j] for j in range(total_dofs)]
            i += total_dofs

    def compute_configuration_matrix(self) -> None:
        super().compute_configuration_matrix()
        self._read_fd()

    def allocate(self, tau: np.ndarray, _d_tau: np.ndarray = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        b_matrix_all = np.concatenate([i.B for i in self._actuators], axis=1)
        w_matrix = np.diag(sum([i.W for i in self._actuators], []))
        b_matrix = b_matrix_all[self.force_torque_components, :]
        tau_vector = tau[self.force_torque_components, :]

        fd = self._fd.reshape(-1, 1)

        f_matrix = self.solve(b_matrix, w_matrix, tau_vector, fd)

        self._command(f_matrix)
        return f_matrix, None

    def solve(self, b_matrix: np.ndarray, w_matrix: np.ndarray, tau: np.ndarray, fd: np.ndarray):

        # Slack variables act in tau (force-torque) space:
        #   B * f + s = tau  ->  one slack per DOF
        n_dof, n_actuators = np.shape(b_matrix)

        constraint_matrix = np.block([b_matrix, np.identity(n_dof)])

        w_slack = np.eye(n_dof) * 1e2

        cost_matrix = np.block([
            [w_matrix,                       np.zeros((n_actuators, n_dof))],
            [np.zeros((n_dof, n_actuators)), w_slack                       ]
        ])

        P = scipy.sparse.csc_matrix(cost_matrix)

        # Minimise ||f - fd||^2_W  →  linear term is -W @ fd
        q_f = -w_matrix @ fd.flatten()
        q = np.concatenate([q_f, np.zeros(n_dof)])

        A = scipy.sparse.csc_matrix(constraint_matrix)
        b = tau.flatten()

        # Actuator box constraints; slacks are unbounded
        lb = []
        ub = []
        for i in self._actuators:
            limits = i.extra_attributes.get('limits', [-np.inf, np.inf])
            lb.extend([min(limits)] * len(i.W))
            ub.extend([max(limits)] * len(i.W))

        lb.extend([-np.inf] * n_dof)
        ub.extend([np.inf] * n_dof)

        lb = np.array(lb)
        ub = np.array(ub)

        # Solve the QP:
        #   min     0.5 * x' * P * x + q' * x
        #   s.t.    A * x = b,  lb <= x <= ub
        #   where   x = [f; s]
        problem = qpsolvers.Problem(P=P, q=q, A=A, b=b, lb=lb, ub=ub)

        solution = qpsolvers.solve_problem(problem, solver='osqp')

        return solution.x[:n_actuators].reshape(-1, 1)