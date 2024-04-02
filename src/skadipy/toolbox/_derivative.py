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
from abc import ABC, abstractclassmethod, abstractmethod


class DerivativeBase(ABC):

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented()

class ExponentialSmoothing(DerivativeBase):
    """
    Exponential smoothing class
    """

    def __init__(self, r: float = 0.7) -> None:
        """
        Exponential smoothing class constructor

        :param r: A weighting factor in the set [0-1]
        """
        self.r = r
        self.x = None
        self.x_p = None
        self.dx_p = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Exponential smoothing function

        :param x: current value of x
        :return: dx
        """
        if self.x_p is None or self.dx_p is None:
            self.x_p = x
            self.dx_p = x
            return x
        else:
            self.dx_p = ExponentialSmoothing.__func(self.r, x, self.x_p, self.dx_p)
            self.x_p = x
            return self.dx_p

    def reset(self, x: np.ndarray) -> None:
        """
        Reset the exponential smoothing

        :param x: current value of x
        :return:
        """
        self.x_p = x
        self.dx_p = x

    @staticmethod
    def __func(
            r: float, x: np.ndarray, x_p: np.ndarray = None, dx_p: np.ndarray = None
    ) -> np.ndarray:
        r"""
        Exponential smoothing function

        .. math::
            \begin{aligned}
                s_{0}&=x_{0}\\
                s_{t}&=\alpha x_{t}+(1-\alpha )s_{t-1},\quad t>0
            \end{aligned}


        :param r: A weighting factor in the set [0-1]
        :param x: current value of x
        :param x_p: previous value of x
        :param dx_p: last computed filtered x value
        :return:
        """
        if x_p is None or dx_p is None:
            return x
        else:
            return (1 - r) * dx_p + r * (x - x_p)
