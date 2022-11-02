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
    Two dimensional Exponential surface brightness profile.

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
        """Two dimensional Exponential profile function."""

        if cls._gammaincinv is None:
            from scipy.special import gammaincinv
            cls._gammaincinv = gammaincinv

        a, b = h, (1 - ellip) * h
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-(z))
    
    
# =============================================================================

class Ferrers2D(Fittable2DModel):
    r"""
    Two dimensional Ferrers surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Central intensity of the bar
    a_bar : float
        Bar radius
    n_bar: float
        Describe the shape of the profile
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

        I(x,y) = I(r) = I_0_bar\exp\left{1-\left(\frac{r_bar}{a_bar}\right)**2\right}**(n_bar+0.5)

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    .. [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
    """

    amplitude = Parameter(default=1, description="Central intensity of the bar")
    a_bar = Parameter(default=1, description="Bar radius")
    n_bar = Parameter(default=1, description="Describe the shape of the profile")
    x_0 = Parameter(default=0, description="X position of the center")
    y_0 = Parameter(default=0, description="Y position of the center")
    ellip = Parameter(default=0, description="Ellipticity")
    theta = Parameter(default=0.0, description=("Rotation angle either as a "
                                                "float (in radians) or a "
                                                "|Quantity| angle"))
    _gammaincinv = None

    @classmethod
    def evaluate(cls, x, y, amplitude,a_bar, n_bar, x_0, y_0, ellip, theta):
        """Two dimensional Ferrers profile function."""

        if cls._gammaincinv is None:
            from scipy.special import gammaincinv
            cls._gammaincinv = gammaincinv

        a, b = a_bar, (1 - ellip) * a_bar
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(1-(z)**2)**(n_bar+0.5)

