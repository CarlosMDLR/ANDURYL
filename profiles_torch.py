# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:15:27 2022

@author: Usuario
"""

import numpy as np
from astropy import units as u
from astropy.units import Quantity, UnitsError
import torch
from astropy.modeling.core import Fittable1DModel, Fittable2DModel
from astropy.modeling.parameters import InputParameterError, Parameter
from astropy.modeling.utils import ellipse_extent
from scipy.special import gammaincinv

def sersic(n,theta,r_eff,ellip,x,x_0,y,y_0,amplitude):
        bn = (2.0*n) - torch.tensor(0.327)
        theta = theta*np.pi/180
        a, b = r_eff, (1-ellip) * r_eff
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        x_maj = -(x - x_0) * cos_theta - (y - y_0) * sin_theta

        z = torch.sqrt(((x_min / a)**2) + ((x_maj/b)**2))
        return(amplitude * torch.exp(-bn*(((z**(1/n))) - 1)))
# =============================================================================
class Sersic2D:
    def __init__(self,x,y,amplitude,r_eff,n,x_0,y_0,ellip,theta):
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
        x_0 : float
            x position of the center.
        y_0 : float
            y position of the center.
        ellip : float
            Ellipticity.
        theta : float 
            The rotation angle as an angular quantity
            The rotation angle increases counterclockwise from the positive x axis.

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
        References
        ----------
        .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
        .. [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
        """
        
        self.x = x
        self.y=y
        #Sersic params
        self.amplitude = amplitude
        self.r_eff= r_eff
        self.n=n
        self.x_0=x_0
        self.y_0=y_0
        self.ellip=ellip
        self.theta=theta

    def __call__(self):
        """Two dimensional Sersic profile function."""
 
        map_ser = sersic(self.n, self.theta, self.r_eff, self.ellip, self.x, self.x_0, self.y, self.y_0, self.amplitude)
        
        # =============================================================================
        #  Center oversampling
        # =============================================================================   

        try:
            M=150; N = 150
            l_size =15
            y_new, x_new = torch.meshgrid(torch.linspace(int(self.y_0-l_size),int(self.y_0+l_size),M), torch.linspace(int(self.x_0-l_size),int(self.x_0+l_size),N))
        except ValueError:
            print("Oversampling: no")
            return(map_ser)
        else:
            m, n = map_ser[int(self.y_0-l_size):int(self.y_0+l_size),int(self.x_0-l_size):int(self.x_0+l_size)].shape
            if m!=0 and n!=0 and m==n:
                map_ser_new = sersic(self.n, self.theta, self.r_eff, self.ellip, x_new, self.x_0, y_new, self.y_0, self.amplitude)
                new=map_ser_new.reshape((m,int(M/m),n,int(N/n))).mean(3).mean(1)
               
                map_ser[int(self.y_0-l_size):int(self.y_0+l_size),int(self.x_0-l_size):int(self.x_0+l_size)]=new
                print("Oversampling: yes")
            else:
                map_ser=map_ser
                print("Oversampling: no")
        return(map_ser)
    
# =============================================================================
# Nofunciona
# =============================================================================
        # try:
        #     M=10000; N = 10000
        #     l_size =5
        #     ylsize = float((self.y_0-l_size).detach().numpy())
        #     yrsize = float((self.y_0+l_size).detach().numpy())
        #     xlsize = float((self.x_0-l_size).detach().numpy())
        #     xrsize = float((self.x_0+l_size).detach().numpy())
        #     y_new = torch.linspace(ylsize,yrsize,M)
        #     x_new = torch.linspace(xlsize,xrsize,N)

        # except ValueError:
        #     print("Oversampling: no")
        #     return(map_ser)
        # else: 
        #     try:
        #         map_ser_new = sersic(self.n, self.theta, self.r_eff, self.ellip, x_new, self.x_0, y_new, self.y_0, self.amplitude)
        #         map_ser_new.reshape(shape=(100,100))
            
        #         new=map_ser_new.reshape((2*l_size,10,2*l_size,10)).mean(3).mean(1)
               
        #         map_ser[int(self.y_0-l_size):int(self.y_0+l_size),int(self.x_0-l_size):int(self.x_0+l_size)]=new
        #         print("Oversampling: yes")
        #     except RuntimeError:
        #         print("Oversampling: no")
        #         return(map_ser)
        #     except ValueError:
        #         print("Oversampling: no")
        #         return(map_ser)
# =============================================================================
class Exponential2D:
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
    def __init__(self,x,y, amplitude, h, x_0, y_0,ellip, theta):
        self.x = x
        self.y=y
        #Sersic params
        self.amplitude = amplitude
        self.h= h
        self.x_0=x_0
        self.y_0=y_0
        self.ellip=ellip
        self.theta=theta

    def __call__(self):
        """Two dimensional Exponential profile function."""
        theta = -self.theta*np.pi/180
        a, b = self.h, (1 - self.ellip) * self.h
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        x_maj = -(self.x - self.x_0) * sin_theta + (self.y - self.y_0) * cos_theta
        x_min = -(self.x - self.x_0) * cos_theta - (self.y - self.y_0) * sin_theta
        z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
        return self.amplitude * torch.exp(-(z)) 
# =============================================================================

class Ferrers2D:
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

    def __init__(self,x,y, amplitude, a_bar,n_bar, x_0, y_0,ellip, theta):
        self.x = x
        self.y=y
        #Sersic params
        self.amplitude = amplitude
        self.a_bar =a_bar
        self.n_bar=n_bar
        self.x_0= x_0
        self.y_0=y_0
        self.ellip=ellip
        self.theta=theta
    

    def __call__(self):
        """Two dimensional Ferrers profile function."""

        theta = -self.theta*np.pi/180
        a, b = self.a_bar, (1 - self.ellip) * self.a_bar
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        x_maj = -(self.x - self.x_0) * sin_theta + (self.y - self.y_0) * cos_theta
        x_min = -(self.x - self.x_0) * cos_theta - (self.y - self.y_0) * sin_theta
        z = torch.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return self.amplitude * torch.exp(1-(z)**2)**(self.n_bar+0.5)