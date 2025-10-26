from random import sample
from unittest import skip
import UV
import numpy as np
import matplotlib.pyplot as plt

class MCMCObject:
    def __init__(self,catalog,corrCat=None):
        """
        In this class I will manipulate all the photometry and get things from it for each source
        
        Args:
            catalog: catalog with binospec coords and Candels photometry
        """
        self.cat,self.corr      =   catalog,corrCat
        self.mu,self.dmu = 1,0 #No magnification, this is used to produce a Normal distribution mean=1 and std=0
        self.filterNames        =   None
        self.centralWavelengths =   None
        self.FilterEffWidth     =   None
        self.Muv,self.MuvErr    =   None,None
        self.Slope,self.SlopeErr    =   None,None
        self.Luv,self.Dust,self.LHa =   None,None,None
        self.group                   = None
        self.Xion,self.XionErr  =None,None
        self.EWHa,self.EWHaErr=None,None

        if np.isnan(catalog["zsys"])==False:
            self.redshift=catalog["photoz"]
            self.type="specz"  
        elif np.isnan(catalog["z"])==False:
            self.redshift=catalog["z"]      
            self.type="specz" 
        elif np.isnan(catalog["photoz"])==False:
            self.redshift=catalog["photoz"]  
            self.type="photz" 
        print(self.type)     


    def setFilters(self,filterNames,filterErrNames):
        self.filterNames=np.array(list(filterNames)+list(filterErrNames))
        return

    def setCWave(self,centralW):
        self.centralWavelengths=np.array(list(centralW))
        return

    def setEffWidth(self,effW):
        self.FilterEffWidth=np.array(list(effW))
        return

    def setHaFlux(self,HaFlux,HaFluxErr):
        if HaFlux<0.0:
            self.HaFlux=np.nan
            self.HaFluxErr=np.nan
        self.HaFlux=HaFlux
        self.HaFluxErr=HaFluxErr   
        return 

    def setOptCont(self,optCont,optCOntErr):
        self.optCont=optCont
        self.optContErr=optCOntErr   
        return 

    def giveCat(self):
        return self.cat

    def calculateUVslope(self,ShowPlots=False):
        #Right now, the codes fro specz and photz are the same, before I was doing different treatments for each but it made no difference and took longer
        #I would assume photz= normal distribution sigma=0.5, and draw in each MCMC iteration

        if self.type=="specz": #for galaxies with specz
            #I creat a mask for bands that are outside of what I define as the UV continuum 
            #Then double the mask, since we have fluxs and errfluxs
            maskBlueFilters =   UV.removeBands(self.centralWavelengths,self.FilterEffWidth,self.redshift,self.cat,self.filterNames,ShowPlots=ShowPlots) 
            maskForFilters=np.array(list(maskBlueFilters)+list(maskBlueFilters))

            #Here is the code that Runs the MCMC
            #Returns slope,Muv, and errors, as well as other MCMC values, and the chains.
            
            slope,Muv,errs,fluxStuff,pairs=UV.FitPowerLaw(self.cat,list(self.filterNames[maskForFilters]),list(self.centralWavelengths[maskBlueFilters])\
                ,self.corr,self.redshift,self.FilterEffWidth,self.type,self.mu,self.dmu,ShowPlots=ShowPlots)
            self.Slope=slope
            self.Muv=Muv
            self.SlopeErr=errs[0]
            self.MuvErr=errs[1]
            self.group=np.array(pairs)
            
            return [self.Slope,self.SlopeErr],[self.Muv,self.MuvErr],np.array(pairs)

        #This one is the same 
        if self.type=="photz": #For galaxies with photz
            maskBlueFilters =   UV.removeBands(self.centralWavelengths,self.FilterEffWidth,self.redshift,self.cat,self.filterNames,ShowPlots=ShowPlots)
            maskForFilters=np.array(list(maskBlueFilters)+list(maskBlueFilters))

            slope,Muv,errs,fluxStuff,pairs=UV.FitPowerLaw(self.cat,list(self.filterNames[maskForFilters]),list(self.centralWavelengths[maskBlueFilters])\
                ,self.corr,self.redshift,self.FilterEffWidth,self.type,self.mu,self.dmu,ShowPlots=ShowPlots)
            self.Slope=slope
            self.Muv=Muv
            self.SlopeErr=errs[0]
            self.MuvErr=errs[1]
            self.group=np.array(pairs)
            return [self.Slope,self.SlopeErr],[self.Muv,self.MuvErr],np.array(pairs)        
    

    #####
    #Here and below, we dont use for this work

    def calculateEWHa(self):
        if self.HaFlux==-99.0 or self.optCont==-99.0:
            return np.nan,np.nan
        EWHa,dEWHa=UV.GetEW(self.HaFlux,self.optCont,[self.HaFluxErr,self.optContErr])
        return EWHa/(1+self.redshift),dEWHa/(1+self.redshift)

    def calculateEWOiiiHb(self):
        if self.Oiii_HbFlux==-99.0 or self.OHboptCont==-99.0:
            return np.nan,np.nan
        EW,dEW=UV.GetEW(self.Oiii_HbFlux,self.OHboptCont,[self.Oiii_HbFluxErr,self.OHboptContErr])
        return EW/(1+self.redshift),dEW/(1+self.redshift)

    def calculateDustLaw(self):
        Dust=UV.GetDustLaw(self.Slope)
        self.Dust=Dust
        return Dust

    def calculateLuv(self):
        Luv=UV.GetLuminosity(self.redshift,self.Muv)
        self.Luv=Luv
        return Luv

    def calculateLHa(self):
        if self.HaFlux==-99.0:  
            self.LHa=np.nan      
            return np.nan
        LHa=UV.GetLuminosityHa(self.redshift,self.HaFlux)
        self.LHa=LHa
        return LHa

    def calculateIntrinsicLuv(self):
        return self.Luv/self.Dust

    def calculateXion(self):
        if self.HaFlux<0:        
            self.Xion=np.nan
            self.XionErr=np.nan
            return [np.nan,np.nan,np.nan]
        randomHa    =   np.random.normal(self.HaFlux,self.HaFluxErr,len(np.array(self.group)[:,0]))
        sampleSlopes = np.array(self.group)[:,0]
        sampleMuvs   = np.array(self.group)[:,1]
        Xs=[]
        for i in range(0,len(np.array(self.group)[:,0])):
            if randomHa[i]<=0:
                continue
            randomLHa    = UV.GetLuminosityHa(self.redshift,randomHa[i])
            randomLuv    = UV.GetLuminosity(self.redshift,sampleMuvs[i])
            randomDust   = UV.GetDustLaw(sampleSlopes[i])
            #print(sampleSlopes[i],randomDust)
            randomXion=randomLHa/(randomLuv/randomDust)*7.37e11
            Xs.append(randomXion)
        
        Xvals=np.percentile(Xs, [16, 50, 84])

        Xion=Xvals[1]
        XionErr=np.mean([Xvals[1]-Xvals[0],Xvals[2]-Xvals[1]])
        self.Xion=Xion
        self.XionErr=XionErr

        return Xvals

    def calculateSSFR(self):
        mass=10**self.Mass
        sfr=self.SFR
        lmass,umass=10**self.MassErr[0],10**self.MassErr[1]
        lsfr,usfr=self.SFRErr[0],self.SFRErr[1]   
        dmass=np.mean([mass-lmass,umass-mass])
        dsfr=np.mean([sfr-lsfr,usfr-sfr])
        sSFR=sfr/mass
        dsSFR=sSFR*np.sqrt( (dmass/mass)**2 + (dsfr/sfr)**2   )
        self.sSFR=sSFR
        self.sSFRErr=dsSFR
        return sSFR*1e9,dsSFR*1e9
