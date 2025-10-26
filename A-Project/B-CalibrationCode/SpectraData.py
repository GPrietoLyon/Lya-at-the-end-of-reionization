import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii
from astropy.stats import sigma_clip
from rescaleSky import rescaleSky

class SpectraData:
    def __init__(self,fits,error):
        """
        Spectra Data builder, in this class we will manipulate the data & error of the spectra
        
        Args:
            fits: spectra fits file from where the data & header are read
            error: error fits file from where the error data is read

            Also searchs for photometry catalogs
        """
        self.rawData = np.array(fits.data)
        self.error = np.array(error.data)
        self.header = fits.header

    def giveData(self):
        #print(self.rawData)
        return self.rawData

    def giveError(self):
        return self.error

    def multScalar(self,scalar):
        """
        Multiply the data and its error by a scalar number
        This is at a self level, so we can later use plotSLit or other methods with the already altered data

        Args:
            scalar: a scalar number to multiply
        """
        multData = scalar * self.rawData
        self.rawData = multData
        multErr = scalar * self.error
        self.error = multErr
        return multData,multErr

    def sumScalar(self,scalar):
        """
        Add a scalar number to the data and its error
        This is at a self level, so we can later use plotSLit or other methods with the already altered data

        Args:
            scalar: a scalar number to add
        """
        sumData = scalar + self.rawData
        self.rawData = sumData
        sumErr = scalar + self.error
        self.error = sumErr
        return sumData,sumErr

    def cut(self,rows,cols): # adapt to add the error
        """
        Cut a section of the slit given X & Y coordinates [not wavelength]
        This is at a self level, so we can later use plotSLit or other methods with the already altered data
        Args:
            rows: range of rows to keep (ex. [20,30])
            cols: range of columns to keep (ex. [100,200])
        """
        cutData=[]
        numRows=len(self.rawData)
        numCols=len(self.rawData[0])

        for i in range(0,numRows):
            if rows[0]<=i<=rows[1]:
                cutData.append([self.rawData[i][j] for j in range(0,numCols) if cols[0]<=j<=cols[1]])
        self.rawData=cutData
        return cutData


    def plotSlit(self):
        """
        Plots the slit
        """
        plt.imshow(self.rawData)
        plt.show()





    def SigmaClipping(self,SigValue=10,IterNum=5):

        """
        Most cosmic rays are labeled as nans and can be discarded later when merging, but some of them have high fluxes that need to be discarded with sigma clipping
        
        Args:

            SigValue : Clipping threshold
            IterNum : Number of iterations of the clipping
            
        Returns:
            Updates self.rawData                                 
        """
        #print(np.nanstd(self.rawData))
        #print(np.nanmedian(self.rawData))
        self.rawData = sigma_clip(self.rawData,sigma=SigValue,maxiters=IterNum,masked=False,axis=1)



    def collapse(self, ywidth=5, dat_type='flux',YPIX="a"):
        """
        Extract 1D spectrum using a boxcar (rectangular) aperture
        
        Args:

            ywidth: Y pixel width of aperture
            
        Returns:
            dat1D/err1D: 1D flux density or flux density error array in '1E-17 erg/cm^2/s/Angstrom' divided by the corresponding error                                       
        """
        if YPIX=="a":
            YPIX=self.header['SLITYPIX']

        dat2D = self.rawData
        ypos = int(YPIX)
        err2D = self.error
        dat1D = np.nansum(dat2D[ypos-int(ywidth/2):ypos+int(ywidth/2)], axis=0)
        err1D = np.sqrt(np.nansum(err2D[ypos-int(ywidth/2):ypos+int(ywidth/2)]**2., axis=0))
        return dat1D,err1D

    def skySpectra(self,cont_size=4):
        """
       Extract 1D sky spectrum, ignoring the edges of the slit
        
        Args:

            ExtraSource: In case the slit contains multiple sources, this will be the Ypix of the continuum
            
        Returns:
        
            Sky1D: Collapsed 2D sky spectra into a 1D array
                                   
        """                                 
        #Sky1D=[]
         #Size of the continuum that we wont take 
        #for i in range(0,len(self.rawData)):
        #    if (i<np.round(self.header['SLITYPIX'])-cont_size or i>np.round(self.header['SLITYPIX'])+cont_size):
        #        Sky1D.append(self.rawData[i])

        #Cut the borders
        Sky1D=self.rawData[3:-3,:]
        return np.nanmedian(Sky1D,axis=0)


    def emptyFlux(self,ExtraSource=9999):
        """
        Obtain median flux of the empty regions in the slit, we use 5 regions where there are no skylines, and dont consider rows near source
        
        Args:

            ExtraSource: In case the slit contains multiple sources, this will be the Ypix of the continuum
            
        Returns:
            emptyFlux: median flux of empty regions                                    
        """    
        regions=[[2110,2418],[2831,3208],[3665,4052],[4541,4589],[4620,4688]] # in pixels
        emptyFlux=[]
        cont_size=4 #Size of the continuum that we wont take 
        for r in regions:
            for row in range(0,len(self.rawData)):
                if (row<np.round(self.header['SLITYPIX'])-cont_size or row>np.round(self.header['SLITYPIX'])+cont_size) and (row<np.round(ExtraSource)-cont_size or row>np.round(ExtraSource)+cont_size) and (row>4 and row<len(self.rawData)-4):
                    for col in range(0,len(self.rawData[0])):
                        if r[0]<=col<=r[1]:
                            emptyFlux.append(self.rawData[row][col])
        return np.nanmedian(emptyFlux)

            

    def processSlit(self,skySpectra,skyscale=None):
    
        """
        Removes sky lines from slit
            
        Args: 
        skySpectra: Combined median sky spectra for all the slits in the exposure

        Returns:
            nothing, but changes self to the 2D data with the rescaled sky subtracted                                   
        """  

        self.rawData,values= rescaleSky(skySpectra,self.rawData,self.error,self.header,skyscale=skyscale)
        return values
        #return rescaleSky(skySpectra,self.rawData,self.error)


    def save2fits(self,file,name="nombre",folder=""):
        #Same fit file with same format
        if name=="nombre":
            name=self.header["EXTNAME"]+" "+self.header["SLITOBJ"]
        fitsfile=fits.open(file)
        fitsfile[0].data=self.rawData
        fitsfile[1].data=self.error
        fitsfile.writeto(folder+name+'.fits',overwrite=True)


    def save2fitsNew(self,data,var,ypix=None,name="nombre",folder=""):
        #Makes new Fit file with new format
        if name=="nombre":
            name=self.header["EXTNAME"]+" "+self.header["SLITOBJ"]
        fitsfile=fits.open("Example_Efield.fits")
        example=fits.open('new1.fits')
        fitsfile[0].data=data
        fitsfile[0].header=example[0].header+self.header
        fitsfile[0].header['CUNIT1']='Angstrom'
        fitsfile[0].header['CRVAL1']=6760.0
        fitsfile[0].header['CDELT1']=0.620000004768
        fitsfile[0].header['CD1_1']=0.620000004768
        if ypix!=None:
            fitsfile[0].header["SLITYPIX"]=ypix
        fitsfile[1].data=var

        fitsfile.writeto(folder+name+'.fits',overwrite=True)

        