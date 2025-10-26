import matplotlib.pyplot as plt
import numpy as np
from Tools import *
from scipy import integrate
import scipy


class BayesGalaxy:
    def __init__(self,catalog):
        """
        In this class I will save all the data for individual galaxies
        
        Args:
            catalog: catalog with binospec coords and Candels photometry
        """
        self.cat                =   catalog

        if np.isnan(self.cat["LyaFlux"])==True:
            self.type   =   "nonLAE"
        if np.isnan(self.cat["LyaFlux"])==False:
            self.type   =   "LAE"
        self.redshift=self.cat["z"]
        if np.isnan(self.redshift)==True:
            self.redshift=self.cat["photoz"]

        self.Lum            =   None
        self.EW             =   None
        self.dEW             =   None
        self.UVslope        =   None
        self.Muv            =   None
        self.FWHM           =   None
        self.Skew           =   None
        self.dMuv           =None
        self.dSlp           =None 
        self.zs=None



class BayesInf:
        """
        In this class I will save all the data for the whole sample
        
        Args:
            self.priorX : These are just the parameters for the fitted skewed gaussian to the slope, Muv priors, More have to be added here if needed
        """
        def __init__(self):
            self.types          =   None
            self.Lum            =   None
            self.EW            =   None
            self.EW_obs             =   None
            self.dEW             =   None
            self.UVslope        =   None
            self.Muv            =   None
            self.FWHM           =   None
            self.Skew           =   None
            self.priorSlope        =   [None,None,None] 
            self.priorMuv       =      [None,None,None] 
            self.wtab           =   None
            self.dMuv           =None
            self.dSlp           =None 
            self.UV             =None
            self.dUV            =None
            self.zs= None
            self.SNcut= None
            self.noise= None
            self.orisize=None

 

        def GenerateErrorUplims(self,constant=[False,5]):
            if constant[0]==True:
                for i in range(0,len(self.dEW)):
                    if self.types[i]=="nonLAE":
                        self.dEW[i]=constant[1]
            if constant[0]==False:
                for i in range(0,len(self.dEW)):
                    if self.types[i]=="nonLAE":
                        self.dEW[i]=self.EW_obs[i]/constant[1]      

        def RandomDrawEW(self):
            nEW=[]
            for w,dw in zip(self.EW,self.dEW):
                ew=np.random.normal(loc=w,scale=dw)
                nEW.append(ew)

            self.EW_obs=np.array(nEW)

        def GenerateErrors(self,noise=5):
            self.noise=noise
            self.dEW=np.full(len(self.EW), noise)

        def Classify(self,SNcut=5):
            self.SNcut=SNcut
            types=[]
            for ew,dew in zip(self.EW_obs,self.dEW):
                if ew/dew<SNcut:
                    types.append("nonLAE")
                elif ew/dew>=SNcut:
                    types.append("LAE")
            self.types=types


        def GenerateWtab(self):
            self.wtab=np.array([np.linspace(-500.,ew, 1000) for ew in self.EW_obs])


        def GenerateMockData(self,mathParamsA,mathParamsW):
            ew=np.linspace(-500.,500,1001)

            physParams=[[self.Muv[i],self.UVslope[i]] for i in range(0,len(self.Muv))]
            ProbSets=[]
            As,Ws=[],[]
            for pSet in physParams:
                A   =   self.ParameterModel(pSet,mathParamsA)
                Wo  =   self.ParameterModel(pSet,mathParamsW)
                y   =   self.OriginalLikelihood(ew,A,Wo)
                ProbSets.append(y)
                As.append(A)
                Ws.append(Wo)
            return ew,ProbSets,As,Ws
        

        def GenerateMockDataConvolved(self,mathParamsA,mathParamsW):
            ew=np.linspace(-500.,500,1001)
            physParams=[[self.Muv[i],self.UVslope[i]] for i in range(0,len(self.Muv))]

            #print(self.UV,self.dUV) 

            ProbSets=[]
            As,Ws=[],[]
            i=0
            for pSet in physParams:


                A   =   self.ParameterModel(pSet,mathParamsA)
                Wo  =   self.ParameterModel(pSet,mathParamsW)
                y   =   self.Likelihood(ew,self.dEW[i],A,Wo)
                ProbSets.append(y)
                As.append(A)
                Ws.append(Wo)
                i=i+1
            return ew,ProbSets,As,Ws



        def OriginalLikelihood(self,ew,A,Wo):
            """
            Likelihood? : Probability of having a certain value of EW given the exponential parameters A and Wo

            Args:
                ew: Equivalent widths (our values?)
                A : Parameter exponential (should depend on our other observables, such as uvslope and muv)
                Wo : Parameter exponential (should depend on our other observables, such as uvslope and muv)
            """

            if Wo<0:
                A=0

            return ((A/Wo)*np.exp(-ew/Wo)*np.heaviside(ew,0.0) + (1-A)*DeltaDirac(ew) )


        def Likelihood(self,ew,dEW,A,Wo):
            """
            Likelihood? : Probability of having a certain value of EW given the exponential parameters A and Wo

            Args:
                ew: Equivalent widths (our values?)
                A : Parameter exponential (should depend on our other observables, such as uvslope and muv)
                Wo : Parameter exponential (should depend on our other observables, such as uvslope and muv)
            """
            if Wo<0:
                A=0
    
            dEW_2 = dEW**2.
            p1 = (1. - A) * np.exp(-0.5 * ew**2./dEW_2) / np.sqrt(2.*np.pi) / dEW
            X  = (dEW_2/Wo - ew) / np.sqrt(2.) / dEW
            p2 = 0.5 * A / Wo * np.exp(0.5*(dEW_2 - 2*ew*Wo)/Wo**2.)*scipy.special.erfc(X)
            p = p1 + p2
            return p

            #return ((A/Wo)*np.exp(-ew/Wo)*np.heaviside(ew,0.0) + (1-A)*DeltaDirac(ew) )
        
            
        def ParameterModel(self,params,cts):
            """
            I define the parameters for the exponential model, which are A and Wo, they both have the same shape, so I just
            use one function. I do not know what to input into this right now...

            See page 15 of Oyarzun 2017. I printed it
            Args:
                params: this are the observable parameters that we input such as Muv,Uvslope... might add more later
                cts : the constants that accompany each param.... this are the values that we have to find, I think
            """
            ####for clarity   
            pUV,pSlope    =   params[0],params[1]
            cUV,cSlope,cCte    =   cts[0],cts[1],cts[2]
            ########

            Parameter   =   (cUV*pUV)   +   (cSlope*pSlope)     +   cCte 
            return Parameter

        def LikNoDet(self,wtab,dew,A,Wo):
             like = self.Likelihood(wtab,dew,A,Wo)
             return integrate.trapz(like,wtab)

        def Posterior(self,ew,dew,physParams,mathParamsA,mathParamsW):

            A=self.ParameterModel(physParams,mathParamsA)
            Wo=self.ParameterModel(physParams,mathParamsW)

            #probLike=self.Likelihood(ew,A,Wo)
            probLike=[]
            for i in range(0,len(self.types)):
                if self.types[i]=="LAE":
                    p   =   self.Likelihood(ew[i],dew[i],A[i],Wo[i])
                if self.types[i]=="nonLAE":
                    p   =   self.LikNoDet(self.wtab[i],dew[i],A[i],Wo[i])
                
                if np.isnan(p)==True:
                    continue

                probLike.append(p)
                
            return np.array(probLike)











