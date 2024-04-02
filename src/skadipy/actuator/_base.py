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
from abc import ABC, abstractmethod

import numpy as np
import pyquaternion as quat
from shapely import geometry

from ._type import ConfigurationType


class ActuatorBase(ABC):
    CONFIGURATION_TYPE = ConfigurationType.UNKNOWN

    def __init__(
        self,
        position: geometry.Point = geometry.Point(),
        orientation: quat.Quaternion = quat.Quaternion(),
        weight: float = float(1.0),
        extra_attributes: typing.Dict[str, float] = { 'rate_limit': None, 'saturation_limit': None },
    ) -> None:
        """
        Base actuator.

        @type position: geometry.Point
        @param position: The position of the actuator.
        @type orientation: quat.Quaternion
        @param orientation: The orientation of the actuator.
        @type weight: float
        @param weight: Weight of the actuator. Default is 1.0

        @rtype: None
        @return: None
        """

        # The position of the actuator.
        self.position = position

        # The orientation of the actuator.
        self.orientation = orientation

        # The allocation configuration matrix.
        self.B = None

        # The weight of the actuator.
        # This can be a scalar or a vector. Depending on the actuator. If the
        # actuator has multiple degrees of freedom, the weight is a vector.
        # This is decided by the subclass's compute_allocation_configuration
        # method.
        self.W = [weight]

        self.force = np.empty((1, 6), dtype=np.float32)

        self.extra_attributes = extra_attributes

    @abstractmethod
    def compute_contribution_configuration(self) -> None:
        r"""
        Compute the contribution configuration matrix.

        This matrix describes the contribution of the actuator to the
        allocation matrix. This is done in a separate method so that
        subclasses can override it.
        """
        raise NotImplementedError()

    @abstractmethod
    def command(self, force: np.ndarray) -> None:
        r"""
        Command the actuator.

        :param force: The force to command the actuator with.
        """
        self.force = force


    @property
    def weight(self) -> float:
        r"""
        Get the weight of the actuator.

        :return: The weight of the actuator.
        """
        return self.W[0]

    @weight.setter
    def weight(self, value: float) -> None:
        r"""
        Set the weight of the actuator.

        :param value: The weight of the actuator.
        """
        self.W = [value]
