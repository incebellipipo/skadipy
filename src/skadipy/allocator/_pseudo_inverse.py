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
import typing
from ._base import AllocatorBase
from ..toolbox._weighted_pseudo_inverse import weighted_pseudo_inverse

class PseudoInverse(AllocatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get column number of B matrix to init fd
        self._fd = None

    def _read_fd(self) -> None:

        self._fd = np.empty((self._b_matrix.shape[1]), dtype=np.float32)

        i = 0
        for a in self._actuators:
            total_dofs = a.B.shape[1]

            a_fd = a.extra_attributes.get('desired_force', [0.0] * total_dofs)

            desired_force = np.array(
                [a_fd[i] for i in range(total_dofs)],
                dtype=np.float32
            )
            self._fd[i:i + total_dofs] = desired_force
            i += total_dofs


    def compute_configuration_matrix(self) -> None:
        """
        Compute the configuration matrix

        This method is implemented by the subclasses.
        """
        super().compute_configuration_matrix()

        self._read_fd()

    def allocate(self, tau: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
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

        # Compute the weighted pseudo-inverse
        b_w_pinv = weighted_pseudo_inverse(b_matrix, w_matrix)

        Q_w = np.eye(w_matrix.shape[0], dtype=np.float32) - b_matrix.T @ b_matrix

        fd = np.reshape(self._fd, (self._fd.shape[0], 1))

        # Compute the force matrix
        f_matrix = b_w_pinv @ tau[self.force_torque_components, :] +  Q_w @ fd

        self._command(f_matrix)

        # Return the force matrix
        return f_matrix, None
