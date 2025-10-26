import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Tools import *

def SearchMin(arr):
    """
    Searchs for the minimum of 2d array, but only considering the flux
    
    """
    vals=   [i["Flux"] for i in arr]
    ide =   vals.index(min(vals))
    return arr[ide]



class LimMatrix:
    def __init__(self,matrix,id):
        """
        In this class I will manipulate all the photometry and get things from it for each source
        
        Args:
            catalog: catalog with binospec coords and Candels photometry
        """
        self.matrix                =   matrix
        self.ID                    =    id
        self.FluxLim               =  None

    def SearchLowest(self,arr,SkyWidth=3):
        """
        Look for lowest observed Flux in 2d array 
        """
        lowestFlux=np.nan
        lowestFWHM=np.nan
        lowestRedshift=np.nan
        for data in arr:
            if float(data[0])==1:
                lowestFlux=float(data[2])
                lowestFWHM=float(data[1])
                lowestRedshift=float(data[3])
                return {"Flux":lowestFlux,"FWHM":lowestFWHM,"Redshift":lowestRedshift}

        
        return {"Flux":lowestFlux,"FWHM":lowestFWHM,"Redshift":lowestRedshift}

    def UpperFlux(self,b=20,fluxes=None,doFit=False):
        plt.style.use("seaborn-poster")
        plt.rcParams["figure.figsize"] = (10,5)
        """
        Run This to retrieve the FluxLimits and save them to self.FluxLim
        
        Args:
            none
        """
        if type(fluxes)==type(None):
            lowestIngal=[]
            for j in self.matrix:
                for i in j[:]:
                    l=self.SearchLowest(i)
                    lowestIngal.append(l)
            fluxes=np.array([l["Flux"] for l in lowestIngal])*1e17

        try:
            counts,bins,__=plt.hist(fluxes,bins=b)
        except:
            self.FluxLim=np.nan
            return self.FluxLim
        
        if doFit == True:
            #fwhm=[l["FWHM"] for l in lowestIngal]

            plt.xlabel("Flux Limit")

            bins=np.array([(bins[i]+bins[i+1])/2 for i in range(0,len(bins)-1)])
            bins=bins[counts>0]
            counts=counts[counts>0]
            nll = lambda *args: -log_likelihood_ML(*args)   #ML fit
            initial = np.array([500,1.5,1,10 ])
            soln = minimize(nll, initial, args=(bins,counts, counts))
            flux,mu,sig,C = soln.x
            print(flux,mu,sig,C)
            x=np.linspace(bins[0],bins[-1],10000)
            plt.plot(bins,counts,".",color="red")
            plt.plot(x,gaussian_ML(x,flux,mu,sig,C))

            self.FluxLim = mu*1e-1

        else:
            plt.axvline(x=np.nanmedian(fluxes),ls="--",color="purple")

            self.FluxLim    =   np.nanmedian(fluxes)
        plt.xlabel("Flux Limits [erg/s-1/cm-2/A]")
        plt.show()     
        return self.FluxLim



"""
def SearchLowest(self,arr,StrongSky,SkyWidth=3):

    Look for lowest observed Flux in 2d array 

    lowestFlux=np.nan
    lowestFWHM=np.nan
    lowestRedshift=np.nan
    for data in arr:
        for SS in StrongSky:
            SShigh=WaveToRedshift(SS+SkyWidth)
            SSlow=WaveToRedshift(SS-SkyWidth)
            if float(data[0])==1 and (float(data[3])>float(SShigh) or float(data[3])<float(SSlow)):
                lowestFlux=float(data[2])
                lowestFWHM=float(data[1])
                lowestRedshift=float(data[3])
                return {"Flux":lowestFlux,"FWHM":lowestFWHM,"Redshift":lowestRedshift}

    
    return {"Flux":lowestFlux,"FWHM":lowestFWHM,"Redshift":lowestRedshift}
"""