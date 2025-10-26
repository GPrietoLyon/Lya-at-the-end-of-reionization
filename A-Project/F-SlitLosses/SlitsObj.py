import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import quad
from LyaSurfaceBrightness import *

class Slits:
    def __init__(self,HDU,mask_observed):
        self.HDU = HDU
        self.mask_observed=mask_observed
        self.slitHeight = HDU[0].header["SLITHEIG"]*u.arcsec
        #self.header = header

    def giveYpix(self,Ypix):
        #print(self.rawData)
        self.Ypix=Ypix

    def add_cat(self,Catalog):
        Catalog=Catalog[Catalog["id_charlotte"]==self.HDU[0].header["SLITOBJ"]]
        self.cat=Catalog
        self.redshift=Catalog["z"]

    def slit_physical_size(self):
        """
        Desription: 
        I get the slit size in arcseconds, and return it in physical Kpc
        """
        #Define Cosmology
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        #Size of the slit in arcsec
        size_arcsec=self.slitHeight
        d_A = cosmo.angular_diameter_distance(z=self.redshift).to(u.kpc)
        self.slit_size_kpc=(size_arcsec*d_A).to(u.kpc, u.dimensionless_angles())[0]

    def kpc_to_arcsec(self,size_kpc):
        """
        Desription: 
        Take a size in Kpc and transform it to arcsec, depending on redshift
        and cosmology
        """
        #Define Cosmology
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        #Physical size
        size_kpc=size_kpc*u.kpc
        #Angular diameter distance
        d_A = cosmo.angular_diameter_distance(z=self.redshift).to(u.kpc)
        #Result will be in radians, so we convert to arcsec
        size_arcsec=(size_kpc/d_A)*u.rad.to(u.arcsec)
        self.gal_size_arcsec=size_arcsec
        return self.gal_size_arcsec


    def convolve_galaxy_with_psf(self,mask_psf_observed,show_plot=False):
        shape = (101,101) 
        self.galaxy_kernel=shape
        mask=self.mask_observed
        sigma_x = mask_psf_observed[mask]/2.355       # Example standard deviation in the x-direction
        sigma_y = mask_psf_observed[mask]/2.355         # Example standard deviation in the y-direction

        # Generate the 2D Gaussian array
        galaxy_model,galaxy_x,galay_y = generate_exponential_circle(shape,"core",self.kpc_to_arcsec)
        galaxy_model=galaxy_model+generate_exponential_circle(shape,"halo",self.kpc_to_arcsec)[0]
        PSF_array,PSF_x,PSF_y = gaussian_2d(shape, 1, sigma_x, sigma_y)
        # Convolve the galaxy model with the PSF
        convolved_galaxy = convolve_galaxy_with_psf(galaxy_model, PSF_array)

        if show_plot==True:       
            plt.pcolor(galaxy_x,galay_y,galaxy_model,vmin=0,vmax=1)
            plt.xlabel("Arcsec")
            plt.ylabel("Arcsec")
            plt.title("Galaxy Model")
            plt.xlim(-2.5,2.5)
            plt.ylim(-2.5,2.5)
            plt.show()
            plt.pcolor(PSF_x,PSF_y,PSF_array,vmin=0,vmax=0.01)
            plt.title("PSF")
            plt.xlabel("Arcsec")
            plt.ylabel("Arcsec")
            plt.xlim(-2.5,2.5)
            plt.ylim(-2.5,2.5)
            plt.show()
            plt.pcolor(galaxy_x,galay_y,convolved_galaxy,vmin=0,vmax=1)
            plt.title("Convolved Galaxy")
            plt.xlabel("Arcsec")
            plt.ylabel("Arcsec")
            plt.xlim(-2.5,2.5)
            plt.ylim(-2.5,2.5)
            plt.show()

        self.convolved_galaxy_2d=convolved_galaxy
        self.convolved_galaxy_axis=galaxy_x
        return convolved_galaxy

    
    def cut_galaxy_in_1d(self,show_plot=False):
        middle_kernel=int((self.galaxy_kernel[0]-1)/2)
        slice_galaxy=self.convolved_galaxy_2d[middle_kernel]

        half_slice_galaxy=slice_galaxy[middle_kernel:]
        half_slice_axis=self.convolved_galaxy_axis[middle_kernel:]
        if show_plot==True:
            plt.plot(half_slice_axis,half_slice_galaxy)
            plt.show()
        
        self.galaxy_1d=half_slice_galaxy
        self.axis_1d=half_slice_axis

    def slit_loss_calculate(self,extraction_size=5,show_plot=False):
        """
        Calculates the slit loss by Integrating the 1D profile, then integrating the same 1D profile
        between 0,extraction_size/2
        
        Parameters:
            extraction_size : The size where we extracted the spectra from, in arcsec
            
        """


        full_galaxy=np.trapz(self.galaxy_1d, dx=self.axis_1d[1]-self.axis_1d[0])
        integrand = interp1d(self.axis_1d,self.galaxy_1d, kind='linear')
        extraction_galaxy, _ = quad(integrand,0, extraction_size/2)
        self.slit_loss=extraction_galaxy/full_galaxy

        if show_plot==True:
            plt.title("Slit Loss = "+str(np.round(self.slit_loss,2)))
            plt.plot(self.axis_1d,self.galaxy_1d)
            plt.axvspan(0,extraction_size/2, alpha=0.5, color='gray')
            plt.axvline(x=extraction_size/2,ls="--",color="gray")
            plt.xlabel("Size [arcsec]")
            plt.ylabel("Normalized Surface brightness [1/arcsec]")
            plt.ylim(0,0.6)
            plt.show()
        
