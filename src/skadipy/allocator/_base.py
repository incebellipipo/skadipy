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

from abc import ABC, abstractmethod
from enum import IntFlag
import typing

import numpy as np

from .. import actuator


class ForceTorqueComponent(IntFlag):
    X = 0  # Surge
    Y = 1  # Sway
    Z = 2  # Heave
    K = 3  # Roll
    M = 4  # Pitch
    N = 5  # Yaw


class AllocatorBase(ABC):
    def __init__(
        self,
        actuators: typing.List[actuator.ActuatorBase] = (),
        force_torque_components: typing.List[ForceTorqueComponent] = None,
    ) -> None:
        """
        Allocator base class

        :param actuators: List of actuators
        :param force_torque_components: List of force-torque components
        """
        self._actuators = actuators

        self._force_torque_components = force_torque_components

        # Configuration matrix
        self._b_matrix = np.empty((6, 0), dtype=np.float32)

        # Weight matrix
        self._w_matrix = np.empty((0, 0), dtype=np.float32)

        self._allocated = np.empty((0, 0), dtype=np.float32)

        self.compute_configuration_matrix()

    @property
    def force_torque_components(self) -> typing.List[ForceTorqueComponent]:
        """
        Get the force-torque components with the type of ForceTorqueComponent.

        :return: Force-torque components
        """
        return self._force_torque_components

    @force_torque_components.setter
    def force_torque_components(self, components: typing.List[ForceTorqueComponent]) -> None:
        """
        Set the force-torque components with the type of ForceTorqueComponent.
        """
        self._force_torque_components = components

    def add_actuator(self, actuator: actuator.ActuatorBase) -> None:
        """
        Add an actuator to the allocator

        :param actuator: Actuator to append
        :return: None
        """
        self._actuators.append(actuator)

    def compute_configuration_matrix(self) -> None:
        """
        Compute the configuration matrix

        This method calls the compute_contribution_configuration method of each
        actuator and concatenates the contribution configuration matrices to
        create the configuration matrix.

        :return: None
        """
        # Compute capability configuration
        for actuator in self._actuators:
            # Compute the contribution of the actuator to the configuration matrix
            actuator.compute_contribution_configuration()

        # Concatenate capability configurations to create the B matrix
        self._b_matrix = np.concatenate([i.B for i in self._actuators], axis=1)

        # Compute the weight matrix
        self._w_matrix = np.diag(sum([i.W for i in self._actuators], []))

    @abstractmethod
    def allocate(self, tau: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Allocate control input to actuators

        This method is implemented by the subclasses.

        :param tau: Control input
        :return: Allocation
        """
        raise NotImplementedError()

    @abstractmethod
    def allocate(self, tau: np.ndarray, d_tau: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Allocate control input to actuators

        This method is implemented by the subclasses.

        :param tau: Control input
        :return: Allocation, Theta
        """
        raise NotImplementedError()

    def _command(self, f: np.ndarray) -> None:
        r"""
        Command the actuators

        :param f: Control input in :math:`\mathbb{R}^q`
        :return: None
        """
        # Command the actuators
        i = 0
        for actuator in self._actuators:
            # Get the number of rows and columns of the actuator matrix
            # and command the actuator
            _, cols = actuator.B.shape
            actuator.command(f[i:i + cols])
            i = i + cols

    @property
    def allocated(self) -> np.ndarray:
        """
        Get the allocated control input

        :return: Allocated control input
        """
        f = np.concatenate([i.force for i in self._actuators])
        self._allocated = self._b_matrix @ f
        return self._allocated


