import matplotlib.pyplot as plt
import numpy as np
from Tools import *




class PhotoObject:
    def __init__(self,catalog):
        """
        In this class I will manipulate all the photometry and get things from it for each source
        
        Args:
            catalog: catalog with binospec coords and Candels photometry
        """
        self.cat                =   catalog
        self.filterNames        =   None
        self.centralWavelengths =   None
        self.FilterEffWidth     =   None
        self.UVfitParams        =   None
        self.UVfitErrs          =   None
        self.UVfluxData         =   None
        self.MuvErr             =   None
        self.LyaFlux            =   10**self.cat["LyaFlux"]
        self.dLya           =   10**self.cat["LyaErr"]
        self.EW             =   None
        self.Muv            =   None
        self.dEW            =   None


        if np.isnan(self.cat["LyaFlux"])==True:
            self.type   =   "nonLAE"
        if np.isnan(self.cat["LyaFlux"])==False:
            self.type   =   "LAE"

        self.redshift=self.cat["zsys"][0]
        if np.isnan(self.redshift)==True:
            self.redshift=self.cat["z"][0]
        if np.isnan(self.redshift)==True:
            self.redshift=self.cat["photoz"][0]

    def setFilters(self,filterNames,filterErrNames):
        self.filterNames=np.array(list(filterNames)+list(filterErrNames))
        return

    def setCWave(self,centralW):
        self.centralWavelengths=np.array(list(centralW))
        return

    def setEffWidth(self,effW):
        self.FilterEffWidth=np.array(list(effW))
        return

    def giveCat(self):
        return self.cat

    def giveData(self):
        """
        returns all the data of the catalog
        """
        return list(self.cat)

    def giveCols(self):
        """
        returns all the column names of the catalog
        """
        return list(self.cat.columns)


    def filt(self,filter):
        '''
        return requested photometry of a filter
        If the name doesnt match, it will try to find the string in the real column names
        '''
        try:
            return self.cat[filter]
        except:
            print("Filter ",filter," not Found. Searching....")
            for realName in self.giveCols():
                if filter==realName:
                    return self.cat[realName]
            print("Filter ",filter," not Found.")           

    def SeeHowManyBands(self):
        maskBlueFilters =   removeBands(self.centralWavelengths,self.FilterEffWidth,self.redshift)
        maskForFilters=np.array(list(maskBlueFilters)+list(maskBlueFilters))
        fnames=self.filterNames[maskForFilters]
        nan_mask = np.isnan([self.cat[fname] for fname in fnames])
        return ~nan_mask

    def calculateUVslope(self,ShowPlots=False):
        maskBlueFilters =   removeBands(self.centralWavelengths,self.FilterEffWidth,self.redshift,ShowPlots=ShowPlots)
        maskForFilters=np.array(list(maskBlueFilters)+list(maskBlueFilters))
        slope,intercept,errs,fluxStuff=FitPowerLaw(self.cat,list(self.filterNames[maskForFilters]),list(self.centralWavelengths[maskBlueFilters]),ShowPlots=ShowPlots)
        self.UVfitParams = [slope,intercept]
        self.UVfitErrs  =   errs[0]
        self.UVfluxData =   fluxStuff
        return slope - 2

    
    def calculateMUV(self):
        #Calculate errors
        r = np.random
        mu,sig  =   np.median(self.UVfluxData[0]), np.median(self.UVfluxData[1])
        #cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
        #Mapp,z =   r.normal(mu, sig, 1000),self.redshift
        #dL      =   cosmo.luminosity_distance(z).value
        
        #Mabs=Mapp-5*(np.log10(dL*10**6) - 1)
        self.MuvErr = np.median(sig)  #np.nanpercentile(Mabs,84)-np.nanpercentile(Mabs,16)


        if self.UVfitParams!=None:
            Muv =   GetMuv(self.UVfitParams,self.redshift)
            self.Muv=Muv
            return Muv
        else:
            self.Muv=np.nan
            return np.nan

    def calculateLumLya(self):
        if np.isnan(self.cat['LyaFlux'])==False:    
            Lum =   GetLuminosity(self.cat["LyaFlux"],self.redshift)
            return Lum
        elif np.isnan(self.cat['FluxLim'])==False:
            Lum =   GetLuminosity(self.cat["FluxLim"],self.redshift)
            return Lum          
        else:    
            return np.nan

    def calculateEW(self):
        if np.isnan(self.cat['LyaFlux'])==False:    
            EW =   GetEW(self.cat["LyaFlux"][0],self.UVfitParams,self.redshift)
            self.EW=EW
            return EW

        elif np.isnan(self.cat['FluxLim'])==False:    
            EW =   GetEW(self.cat["FluxLim"][0],self.UVfitParams,self.redshift)
            self.EW=EW
            return EW
        else:
            self.EW=np.nan
            return np.nan


    def getEWError(self):
        b,c     =   self.UVfitParams[0],self.UVfitParams[1]
        continuumWave=1241.5*(1+self.redshift)
        logflux =   c+b*np.log10(continuumWave)

        #########

        cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
        dL      =   cosmo.luminosity_distance(self.redshift).value
        MappUV    =   self.Muv+5*(np.log10(dL*10**6) - 1)


        fluxes=10**(-(MappUV+48.6)/2.5)
        flxMinErr=10**(-(MappUV+self.MuvErr+48.6)/2.5)
        flxPlusErr=10**(-(MappUV-self.MuvErr+48.6)/2.5)
        fluxesErr=np.median([flxPlusErr-fluxes,fluxes-flxMinErr],axis=0)
        ######################
        
        dlogflux  = np.abs(logflux -np.log10(10**logflux +fluxesErr  ))
        #print(fluxes,fluxesErr)
        #print(logflux,dlogflux)

        UVflux    =   (10**logflux)*(2.998e18/(continuumWave**2))
        dUV    =   UVflux - (10**(logflux-dlogflux))*(2.998e18/(continuumWave**2))
        Lya=self.LyaFlux
        dLya=self.dLya
       # print(dLya)
        

        dEW=np.sqrt((dUV/UVflux)**2+(dLya/Lya)**2)*self.EW
        self.dEW=dEW
        return dEW

    def getEWErrorMock(self):
        b,c     =   self.UVfitParams[0],self.UVfitParams[1]
        continuumWave=1241.5*(1+self.redshift)
        logflux =   c+b*np.log10(continuumWave)

        #########

        cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
        dL      =   cosmo.luminosity_distance(self.redshift).value
        MappUV    =   self.Muv+5*(np.log10(dL*10**6) - 1)


        fluxes=10**(-(MappUV+48.6)/2.5)
        flxMinErr=10**(-(MappUV+self.MuvErr+48.6)/2.5)
        flxPlusErr=10**(-(MappUV-self.MuvErr+48.6)/2.5)
        fluxesErr=np.median([flxPlusErr-fluxes,fluxes-flxMinErr],axis=0)
        ######################
        
        dlogflux  = np.abs(logflux -np.log10(10**logflux +fluxesErr  ))
        #print(fluxes,fluxesErr)
        #print(logflux,dlogflux)

        UVflux    =   (10**logflux)*(2.998e18/(continuumWave**2))
        dUV    =   UVflux - (10**(logflux-dlogflux))*(2.998e18/(continuumWave**2))
        
        Lya=self.EW*UVflux*(1+self.redshift)
        #if self.type=="LAE":
        errs=np.load("../Catalogs/ErrorsLAE.npy")
        #if self.type=="nonLAE":
        #    errs=np.load("../Catalogs/Errors.npy")
        dLya=10**np.random.normal(loc=np.nanmedian(errs),scale=np.nanstd(errs))
        

        dEW=np.sqrt((dUV/UVflux)**2+(dLya/Lya)**2)*self.EW
        self.dEW=dEW
        return dEW,UVflux,dUV
