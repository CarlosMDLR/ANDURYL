# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:57:59 2022

@author: Usuario
"""
import numpy as np

from astropy import units as u
from astropy.units import Quantity, UnitsError

from astropy.modeling.core import Fittable1DModel, Fittable2DModel
from astropy.modeling.parameters import InputParameterError, Parameter
from astropy.modeling.utils import ellipse_extent


class Exponential2D(Fittable2DModel):
    r"""
    Two dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Central intensity
    hr_eff : float
        Radius scale length
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity.
    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise from the positive x axis.
    Notes
    -----
    Model formula:

    .. math::

        I(x,y) = I(r) = I_0\exp\left(-\frac{R}{h}\right)

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    .. [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
    """

    amplitude = Parameter(default=1, description="Central intensity")
    h = Parameter(default=1, description="Radius scale length")
    x_0 = Parameter(default=0, description="X position of the center")
    y_0 = Parameter(default=0, description="Y position of the center")
    ellip = Parameter(default=0, description="Ellipticity")
    theta = Parameter(default=0.0, description=("Rotation angle either as a "
                                                "float (in radians) or a "
                                                "|Quantity| angle"))
    _gammaincinv = None

    @classmethod
    def evaluate(cls, x, y, amplitude, h, x_0, y_0, ellip, theta):
        """Two dimensional Sersic profile function."""

        if cls._gammaincinv is None:
            from scipy.special import gammaincinv
            cls._gammaincinv = gammaincinv

        a, b = h, (1 - ellip) * h
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-(z))


    @property
    def input_units(self):
        if self.x_0.unit is None:
            return None
        return {self.inputs[0]: self.x_0.unit,
                self.inputs[1]: self.y_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'r_eff': inputs_unit[self.inputs[0]],
                'theta': u.rad,
                'amplitude': outputs_unit[self.outputs[0]]}