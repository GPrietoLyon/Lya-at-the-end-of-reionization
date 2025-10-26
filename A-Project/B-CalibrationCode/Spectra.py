import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
plt.style.use(['seaborn-ticks'])

class Spectra:
    def __init__(self,fits,Data): #Constructor
        """
        Class Builder, this will build the main Spectra object
        
        Args:
            fits: Fits file of the spectra
            Data: SpectraData object (in the spectra data class we will manipulate the data)
        """
        header=fits.header
        self.ID = str(header["SLITOBJ"])
        self.Ra = str(header["SLITRA"])
        self.Dec = str(header["SLITDEC"])
        self.expTime = header["EXPTIME"]
        self.Ypix   =   header["SLITYPIX"]
        self.Data = Data # this is a class
        self.z  =   self.findRedshift()
        self.header=header

    def findRedshift(self,locationCatalog="/Users/gonzalo/Desktop/Code/Gonzalo_Binospec/A-Project/A-Catalogs/"):
        cat=ascii.read(locationCatalog+"Binospec-Candels.cat")
        redshift=None
        for gal in cat:
            if gal["id_charlotte"]==self.ID:
                redshift=gal["z"]
                if np.isnan(gal["z"])==True:
                    redshift=gal["photoz"]

        return redshift

        



   
    
