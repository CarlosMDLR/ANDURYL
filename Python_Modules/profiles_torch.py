# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:15:27 2022

@author: Usuario
"""

#Import modules
import torch
import numpy as np

# Torch is put to work in double precision
torch.set_default_dtype(torch.float64)

def rebin(a, *args):
    """
    Rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns 
    and 4 rows can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.ceil(np.asarray(shape)/np.asarray(args)).astype(int)
    evList = ['a.reshape('] + \
             [str(args[i])+','+str(factor[i])+',' for i in range(len(args))]+\
             [')'] + ['.mean(%d)'%(i+1) for i in range(0,lenShape)]
    
    return eval(''.join(evList))

def sersic(n,theta,r_eff,ellip,x,x_0,y,y_0,amplitude):
    """ 
    Perform a 2D Sersic model, parameters are explained below
    """
    bn = (0.868*n)-torch.tensor(0.142)
    theta = (theta*np.pi/180)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    x_maj = -(x - x_0) * cos_theta - (y - y_0) * sin_theta

    z = torch.sqrt(((x_min)**2) + ((x_maj/(1-ellip))**2))
    invn = 1/n
    return(amplitude * 10**(-bn*(((z/r_eff)**(invn))-1)))

def expo(theta,r_eff,ellip,x,x_0,y,y_0,amplitude):
    """ 
    Perform a 2D Exponential model, parameters are explained below
    """
    theta = (theta*np.pi/180)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    x_maj = -(x - x_0) * cos_theta - (y - y_0) * sin_theta

    z = torch.sqrt(((x_min)**2) + ((x_maj/(1-ellip))**2))
    
    return(amplitude * torch.exp(-(z/r_eff))) 

# =============================================================================
# =============================================================================
# =============================================================================

class Sersic2D:
    def __init__(self,x,y,amplitude,r_eff,n,x_0,y_0,ellip,theta):
        """
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
            The rotation angle increases counterclockwise 
            from the positive x axis.

        Notes
        -----
        Model formula:
            I(x,y) = I(r) = I_e\exp\left\{
                    -b_n\left[\left(\frac{r}{r_{e}}\right)^{(1/n)}-1\right]
                \right\}

        The constant : b_n is defined such that r_e
        contains half the total
        luminosity, and can be solved for numerically.

        References
        ----------
        [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
        [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
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
 
        map_ser = sersic(self.n, self.theta, self.r_eff, self.ellip, self.x,\
                         self.x_0, self.y, self.y_0, self.amplitude)
        
        # =====================================================================
        #  Center oversampling
        # =====================================================================   
        
        # Here the oversampling of the central zone is carried out due to 
        # the characteristics of the very sharp profiles in the center
        ny,nx=map_ser.shape
        try:

            half_size = 10
            x_left= (torch.round(self.x_0)-half_size)-0.5
            x_right= (torch.round(self.x_0)+half_size)-0.5
            y_left= (torch.round(self.y_0)-half_size)-0.5
            y_right= (torch.round(self.y_0)+half_size)-0.5
           
            sub_x = torch.linspace(x_left.item(), x_right.item(), 201)+0.05
            sub_y = torch.linspace(y_left.item(), y_right.item(), 201)+0.05
            sub_yy,sub_xx= torch.meshgrid(sub_y[:-1], sub_x[:-1])
            
            #the new mesh is built and the sersic profile is adjusted
            submodel = sersic(self.n,self.theta,self.r_eff,self.ellip,sub_xx,\
                              self.x_0,sub_yy,self.y_0,self.amplitude)
            
            # It is refinished back to the size of the original mesh 
            # and that part of the model is replaced.
            rebin_size = 20
            rebinned_submodel = rebin(submodel, rebin_size, rebin_size)
            x_left= int(torch.round(self.x_0)-half_size)
            x_right= int(torch.round(self.x_0)+half_size)
            y_left= int(torch.round(self.y_0)-half_size)
            y_right= int(torch.round(self.y_0)+half_size)
           
            map_ser[y_left:y_right,x_left:x_right] = rebinned_submodel
            
            return(map_ser)
        except:
            return(map_ser)
        return(map_ser)

class Exponential2D:
    """
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
    theta : float 
        The rotation angle in radians. The rotation angle
        increases counterclockwise from the positive x axis.
    Notes
    -----
    Model formula:
        I(x,y) = I(r) = I_0\exp\left(-\frac{R}{h}\right)

    References
    ----------
     [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
     [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
    """
    def __init__(self,x,y, amplitude, r_eff, x_0, y_0,ellip, theta):
        self.x = x
        self.y=y
        #Exponential params
        self.amplitude = amplitude
        self.r_eff= r_eff
        self.x_0=x_0
        self.y_0=y_0
        self.ellip=ellip
        self.theta=theta
    def __call__(self):
        """Two dimensional Exponential profile function."""
        map_ser = expo(self.theta, self.r_eff, self.ellip, self.x, self.x_0,\
                       self.y, self.y_0, self.amplitude)
        
        # =====================================================================
        #  Center oversampling
        # =====================================================================   
        # Here the oversampling of the central zone is carried out due to 
        # the characteristics of the very sharp profiles in the center
        ny,nx=map_ser.shape
        try:

            half_size = 10
            x_left= (torch.round(self.x_0)-half_size)-0.5
            x_right= (torch.round(self.x_0)+half_size)-0.5
            y_left= (torch.round(self.y_0)-half_size)-0.5
            y_right= (torch.round(self.y_0)+half_size)-0.5
           
            sub_x = torch.linspace(x_left.item(), x_right.item(), 201)+0.05
            sub_y = torch.linspace(y_left.item(), y_right.item(), 201)+0.05
            sub_yy,sub_xx= torch.meshgrid(sub_y[:-1], sub_x[:-1])
            
            #the new mesh is built and the exponential profile is adjusted
            submodel = expo(self.theta,self.r_eff,self.ellip,sub_xx,self.x_0,\
                            sub_yy,self.y_0,self.amplitude)
                
            # It is refinished back to the size of the original mesh 
            # and that part of the model is replaced.
            rebin_size = 20
            rebinned_submodel = rebin(submodel, rebin_size, rebin_size)
            x_left= int(torch.round(self.x_0)-half_size)
            x_right= int(torch.round(self.x_0)+half_size)
            y_left= int(torch.round(self.y_0)-half_size)
            y_right= int(torch.round(self.y_0)+half_size)
           
            map_ser[y_left:y_right,x_left:x_right] = rebinned_submodel
            
            return(map_ser)
        except:
            return(map_ser)
        return(map_ser)

# =============================================================================
#  IN PREPARATION 
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
    theta : float
        The rotation angle in radians. The rotation angle
        increases counterclockwise from the positive x axis.
    Notes
    -----
    Model formula:

    I(x,y) = I(r) = I_0_bar\exp\left{1-\left(\frac{r_bar}{a_bar}\right)**2\right}**(n_bar+0.5)

    References
    ----------
    [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    [2] https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
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