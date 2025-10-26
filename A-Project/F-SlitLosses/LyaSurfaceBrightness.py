import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import glob as glob
import SlitsObj
import astropy.units as u
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def decreasing_exponential(r,rs):
    return (1/rs)*np.e**(-r/rs)

def surface_brightness_profile(tipo,unit="kpc",obj=None):
    if unit=="kpc":
        core_or_halo_size={"core":0.3,"halo":3.8}
        distance=np.linspace(0,15,100)
    if unit=="arcsec":
        core_or_halo_size={"core":obj.kpc_to_arcsec(0.3),"halo":obj.kpc_to_arcsec(3.8)}
        distance=np.linspace(0,2.5,100)    
    core_or_halo_contribution={"core":0.35,"halo":0.65}
    return distance,\
        core_or_halo_contribution[tipo]*\
            decreasing_exponential(distance,core_or_halo_size[tipo])


def read_data(location,Cat,mask_observed):
    Cat=Cat[Cat["Type"]=="LAE"]
    data_dir = '../../A-Catalogs/data/'
    fname_data = np.sort(glob.glob(data_dir+location)) #read data
    fname_data = [i for i in fname_data if "3DHST" not in i and "f_" not in i] #takes only LAEs
    mask=[]
    Galaxy_from_catalog=[]
    for gal in fname_data:
        g=gal.split(" ")[-1]
        g=g.split(".")[0]
        if g in Cat["id_charlotte"]:
            mask.append(True)
        else:
            mask.append(False)
    Specs=[]
    for data in fname_data[:]:
        HDU = fits.open(data)
        Specs.append(SlitsObj.Slits(HDU,mask_observed))
    return np.array(Specs)[mask]

def generate_exponential_circle(shape,tipo,kpc_to_arcsec):
    """
    Generate a 2D array representing a decreasing exponential function forming a circle around the center.
    
    Parameters:
        shape (tuple): Shape of the output array (height, width).
        decay_rate (float): Decay rate of the exponential function.
    
    Returns:
        ndarray: 2D array containing the exponential function forming a circle around the center.
    """
    # Core and Halo definitions
    core_or_halo_size={"core":kpc_to_arcsec(0.3),"halo":kpc_to_arcsec(3.8)}
    core_or_halo_contribution={"core":0.35,"halo":0.65}
    
    # Generate grid
    x = (np.arange(shape[1]) - (shape[1] // 2))/10
    y = (np.arange(shape[0]) - (shape[0] // 2))/10

    xx, yy = np.meshgrid(x, y)
    
    # Calculate distance from center
    radius = np.sqrt(xx**2 + yy**2)

    # Calculate exponential function based on distance
    exponential_circle = core_or_halo_contribution[tipo]*\
        decreasing_exponential(radius,core_or_halo_size[tipo])
    
    return exponential_circle,x,y


def gaussian_2d(shape, amplitude, sigma_x, sigma_y):
    """
    Generates a 2D Gaussian array centered in the middle of the shape.
    
    Parameters:
        shape (tuple): Shape of the output array (height, width).
        amplitude (float): Amplitude or peak value of the Gaussian.
        sigma_x (float): Standard deviation of the Gaussian in the x-direction.
        sigma_y (float): Standard deviation of the Gaussian in the y-direction.
    
    Returns:
        ndarray: 2D array containing the Gaussian.
    """
    # Calculate center coordinates
    center_x = 0
    center_y = 0

    # Generate grid
    x = (np.arange(shape[1]) - (shape[1] // 2))/10
    y = (np.arange(shape[0]) - (shape[0] // 2))/10
    x, y = np.meshgrid(x, y)

    
    # Calculate Gaussian
    gaussian = amplitude * np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
    
    return gaussian,x,y

def convolve_galaxy_with_psf(galaxy_model, psf):
    # Normalize PSF
    psf /= np.sum(psf)
    
    # Convolve galaxy model with PSF
    convolved_galaxy = convolve2d(galaxy_model, psf, mode='same', boundary='wrap')
    
    return convolved_galaxy



def find_closest_index(list1, list2):
    closest_indices = []

    for num1 in list1:
        closest_index = min(range(len(list2)), key=lambda i: abs(list2[i] - num1))
        closest_indices.append(closest_index)

    return closest_indices

def gaussian(x, amp, mu, std):
    """Normalized Gaussian function"""
    return amp / (np.sqrt(2 * np.pi) * std) * np.exp(-(x - mu)**2 / (2 * std**2))


def find_fwhm(x, y):
    """Fit a Gaussian to data points (x, y) and return the FWHM"""
    # Initial guess for Gaussian parameters
    amp_guess = max(y)
    cen_guess = x[np.argmax(y)]
    wid_guess = (max(x) - min(x)) / 5

    # Fit the Gaussian curve
    popt, _ = curve_fit(gaussian, x, y, p0=[amp_guess, cen_guess, wid_guess])

    # FWHM calculation
    fwhm = 2.355* popt[2]

    return fwhm,popt


def get_columns_from_selected_wave(waves,f_mask):
    return 1

def obtain_psf_for_mask(f_mask):
    #Waves where I take star PSF
    waves=[7193,7227,7384,7697,8036,8331,8604,8975,9250,9348] 
    arcsec_per_pix=0.24 #0.24''/pix
    FWHM=[]
    for f_m in f_mask:
        f=fits.open(f_m)
        wave2D=np.arange(6760.0,6760.0+5631*0.620000004768,0.620000004768)
        index=find_closest_index(waves,wave2D)
        temp_fwhm=[]
        for w in index:
            flux=np.transpose(f[0].data)[w]
            if np.nanmean(flux)==0:
                continue
            space_axis=np.arange(0,arcsec_per_pix*len(flux),arcsec_per_pix)
            #plt.plot(space_axis,flux)
            fwhm,popt=find_fwhm(space_axis,flux)
            temp_fwhm.append(fwhm)
        FWHM.append(np.mean(np.array(temp_fwhm)))
    return np.mean(np.array(FWHM))

