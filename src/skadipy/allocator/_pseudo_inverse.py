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
import typing
from ._base import AllocatorBase
from ..toolbox._weighted_pseudo_inverse import weighted_pseudo_inverse

class PseudoInverse(AllocatorBase):

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

        # Compute the force matrix
        f_matrix = b_w_pinv @ tau[self.force_torque_components, :]

        self._command(f_matrix)

        # Return the force matrix
        return f_matrix, None
