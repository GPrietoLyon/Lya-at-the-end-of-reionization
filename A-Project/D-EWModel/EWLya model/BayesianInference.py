import matplotlib.pyplot as plt
import numpy as np
from Tools import *
from scipy import integrate
import scipy


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


        def GenerateMockData(self,mathParamsA,mathParamsW):
            ew=np.round( np.linspace(-50, 500, num=5501), decimals=1)


            physParams=self.Muv
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
            ew=np.round( np.linspace(-50, 500, num=5501), decimals=1)
            physParams=self.Muv

            #print(self.UV,self.dUV)

            ProbSets=[]
            As,Ws=[],[]
            for pSet in physParams:


                A   =   self.ParameterModel(pSet,mathParamsA)
                Wo  =   self.ParameterModel(pSet,mathParamsW)
                y   =   self.Likelihood(ew,self.noise,A,Wo)
                ProbSets.append(y)
                As.append(A)
                Ws.append(Wo)
            return ew,ProbSets,As,Ws

        def GenerateMockDataConvolved_real(self,mathParamsA,mathParamsW):
            ew=np.round( np.linspace(-50, 500, num=5501), decimals=1)
            physParams=self.Muv

            #print(self.UV,self.dUV)

            ProbSets=[]
            As,Ws=[],[]
            for i,pSet in enumerate(physParams):


                A   =   self.ParameterModel(pSet,mathParamsA)
                Wo  =   self.ParameterModel(pSet,mathParamsW)
                y   =   self.Likelihood(ew,self.dEW[i],A,Wo)
                ProbSets.append(y)
                As.append(A)
                Ws.append(Wo)
            return ew,ProbSets,As,Ws



        def Likelihood(self,ew,dEW,A,Wo):
            """
            Likelihood? : Probability of having a certain value of EW given the exponential parameters A and Wo

            Args:
                ew: Equivalent widths (our values?)
                A : Parameter exponential (should depend on our other observables, such as uvslope and muv)
                Wo : Parameter exponential (should depend on our other observables, such as uvslope and muv)
            """
    
            dEW_2 = dEW**2.
            p1 = (1. - A) * np.exp(-0.5 * ew**2./dEW_2) / np.sqrt(2.*np.pi) / dEW
            X  = (dEW_2/Wo - ew) / np.sqrt(2.) / dEW
            p2 = 0.5 * A / Wo * np.exp(0.5*(dEW_2 - 2*ew*Wo)/Wo**2.)*scipy.special.erfc(X)
            p = p1 + p2
            return p
        
        def ParameterModel(self,physParams,mathParams):
            """
            Take in the physical parameter and the constants of the model.
            Returns A or W
            """
            Muv=physParams
            cMuv,cte= mathParams[0],mathParams[1]
            Parameter   =   ((Muv+20)*cMuv)  +   cte 
            return Parameter

        def OriginalLikelihood(self,ew,A,Wo):
            return ((A/Wo)*np.exp(-ew/Wo)*np.heaviside(ew,0.0) + (1-A)*DeltaFunc(ew) )


        def RandomDrawEW(self,noise): 
            return [np.random.normal(loc=w,scale=noise) for w in self.EW]

        def Classify(self,SNcut=5):
            self.SNcut=SNcut
            dew=self.dEW
            types=[]
            for ew,dew in zip(self.EW_obs,dew):
                if ew/dew<SNcut:
                    types.append("nonLAE")
                elif ew/dew>=SNcut:
                    types.append("LAE")
            self.types=types
            return types
        
        def GenerateWtab(self):
            start_value = -50
            step = 0.01
            end_value=self.SNcut*self.noise
            num_elements = int((end_value - start_value) / step) + 1
            self.wtab=np.round(np.linspace(start_value, end_value, num_elements),1)
            return np.round(np.linspace(start_value, end_value, num_elements),1)

        def GenerateWtab_real(self):
            start_value = -50
            step = 0.1
            wtabs=[]
            for ew in self.EW_obs:
                end_value=ew
                num_elements = int((end_value - start_value) / step) + 1
                wtabs.append(np.round(np.linspace(start_value, end_value, num_elements),1))
            self.wtab=wtabs
            return wtabs

