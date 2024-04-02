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

from ._base import ActuatorBase
from ._type import ConfigurationType


class Azimuth(ActuatorBase):

    # The configuration type of the azimuth actuator is extended planar.
    CONFIGURATION_TYPE = ConfigurationType.EXTENDED_PLANAR

    def compute_contribution_configuration(self) -> None:
        """
        Compute the allocation configuration matrix.

        @rtype: None
        @return: None
        """

        # The azimuth actuator has a 6x2 allocation configuration matrix.
        self.B = np.empty((6, 2), dtype=np.float32)

        # The first three rows of the allocation configuration matrix are the
        # identity matrix.
        self.B[0:2] = np.eye(2, dtype=np.float32)

        self.B[2:3] = np.zeros((1, 2), dtype=np.float32)

        # The last three rows of the allocation configuration matrix are the
        # cross product matrix of the position vector.
        self.B[3:4] = np.array([0, -self.position.z], dtype=np.float32)
        self.B[4:5] = np.array([self.position.z, 0], dtype=np.float32)
        self.B[5:6] = np.array([-self.position.y, self.position.x], dtype=np.float32)

        # The weight vector is a 2x1 vector with the weight of the actuator.
        # Because the azimuth actuator has three degrees of freedom, the weight
        # vector is repeated three times.
        self.W = [self.W[0]] * 2

    def command(self, force: np.ndarray) -> None:
        """
        Command the azimuth actuator.

        @type force: np.ndarray
        """
        self.force = force
        # print(f"{self.CONFIGURATION_TYPE.name} got {force}")
        return super().command(force)
