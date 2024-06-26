#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy
import scipy.io
import skadipy as mc
from typing import List

import skadipy.allocator
import skadipy.allocator.reference_filters as rf
import skadipy.actuator
import skadipy.actuator._base


def save_mat(
        filename: str,
        inputs: np.ndarray,
        xi: np.ndarray,
        outputs: np.ndarray,
        thruster: skadipy.actuator._base.ActuatorBase,
        thetas: np.ndarray = [],
        rho: np.ndarray = [],
        mu: np.ndarray = [],
        gamma: np.ndarray = [],
        zeta: np.ndarray = [],
        lambda_p: np.ndarray = [],
    ):


    scipy.io.savemat(
        file_name=filename,
        mdict={
            "inputs": inputs,
            "xi": xi,
            "outputs": outputs,
            "theta": thetas,
            "rho": rho,
            "mu": mu,
            "zeta": zeta,
            "gamma": gamma,
            "lambda_p": lambda_p,
            "attributes": thruster.extra_attributes
        }
    )