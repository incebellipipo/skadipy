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
import pyquaternion as quat

from ._base import ActuatorBase
from ._type import ConfigurationType


class Fixed(ActuatorBase):
    CONFIGURATION_TYPE = ConfigurationType.BASIC

    def compute_contribution_configuration(self) -> None:
        """
        Compute the allocation configuration matrix for the fixed actuator.
        """

        # The fixed actuator has a 6x1 allocation configuration matrix.
        self.B = np.empty((6, 1), dtype=np.float32)

        projection = self.orientation.rotate(quat.Quaternion(vector=[1.0, 0.0, 0.0]))

        # The force is the projection of the axis of the actuator onto the
        # all three axes of the body frame.
        force = np.array([projection.axis / np.linalg.norm(projection.axis)]).T

        # The torque is the cross product of the position and the force.
        torque = np.array(
            [
                self.position.y * force[2] - self.position.z * force[1],
                self.position.z * force[0] - self.position.x * force[2],
                self.position.x * force[1] - self.position.y * force[0],
            ]
        )

        # The allocation configuration matrix is the concatenation of the force
        # and the torque.
        self.B = np.concatenate((force, torque))

        self.W = [self.W[0]] * 1


    def command(self, force: np.ndarray) -> None:
        self.force = force
        # print(f"{self.CONFIGURATION_TYPE.name} got {force}")
        return super().command(force)
