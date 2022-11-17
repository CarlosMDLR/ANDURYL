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

# =============================================================================
class Sersic2D(Fittable2DModel):
    r"""
    Two dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Surface brightness at r_eff.
    r_eff : float
        Effective (half-light) radius
    n : float
        Sersic Index.
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

    See Also
    --------
    Gaussian2D, Moffat2D

    Notes
    -----
    Model formula:

    .. math::

        I(x,y) = I(r) = I_e\exp\left\{
                -b_n\left[\left(\frac{r}{r_{e}}\right)^{(1/n)}-1\right]
            \right\}

    The constant :math:`b_n` is defined such that :math:`r_e` contains half the total
    luminosity, and can be solved for numerically.

    .. math::

        \Gamma(2n) = 2\gamma (2n,b_n)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Sersic2D
        import matplotlib.pyplot as plt

        x,y = np.meshgrid(np.arange(100), np.arange(100))

        mod = Sersic2D(amplitude = 1, r_eff = 25, n=4, x_0=50, y_0=50,
                       ellip=.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)


        plt.figure()
        plt.imshow(log_img, origin='lower', interpolation='nearest',
                   vmin=-1, vmax=2)
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar()
        cbar.set_label('Log Brightness', rotation=270, labelpad=25)
        cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    .. [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
    """

    amplitude = Parameter(default=1, description="Surface brightness at r_eff")
    r_eff = Parameter(default=1, description="Effective (half-light) radius")
    n = Parameter(default=4, description="Sersic Index")
    x_0 = Parameter(default=0, description="X position of the center")
    y_0 = Parameter(default=0, description="Y position of the center")
    ellip = Parameter(default=0, description="Ellipticity")
    theta = Parameter(default=0.0, description=("Rotation angle either as a "
                                                "float (in radians) or a "
                                                "|Quantity| angle"))
    _gammaincinv = None

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
        """Two dimensional Sersic profile function."""

        if cls._gammaincinv is None:
            from scipy.special import gammaincinv
            cls._gammaincinv = gammaincinv

        bn = cls._gammaincinv(2. * n, 0.5)
        theta = -theta*np.pi/180
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        x_min = -(x - x_0) * cos_theta - (y - y_0) * sin_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-bn * (z ** (1 / n) - 1))
# =============================================================================
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
        theta = -theta*np.pi/180
        a, b = h, (1 - ellip) * h
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        x_min = -(x - x_0) * cos_theta - (y - y_0) * sin_theta
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
        theta = -theta*np.pi/180
        a, b = a_bar, (1 - ellip) * a_bar
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        x_min = -(x - x_0) * cos_theta - (y - y_0) * sin_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(1-(z)**2)**(n_bar+0.5)

