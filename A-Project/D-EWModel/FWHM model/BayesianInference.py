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
            self.typesCat          =   None
            self.Lum            =   None
            self.EW            =   None
            self.fesc          =None
            self.dfesc          =None
            self.fesc_obs          =None
            self.EW_obs             =   None
            self.dEW             =   None
            self.UVslope        =   None
            self.Muv            =   None
            self.FWHM           =   None
            self.dFWHM           =   None
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


        def GenerateMockData(self,A,Wo):
            fesc= np.linspace(-0.5, 3, num=201)

            y   =   self.OriginalLikelihood(fesc,A,10**Wo)
            
            return fesc,y,A,Wo
        

        def GenerateMockData2(self,mathParamsW):
            fesc=np.linspace(-0.5, 1.5, num=201)

            physParams=np.transpose([self.Muv,self.Beta])
            ProbSets=[]
            Ws=[]
            for pSet in physParams:
                Wo  =   self.ParameterModel(pSet,mathParamsW)
                y   =   self.OriginalLikelihood2(fesc,Wo)
                ProbSets.append(y)
                Ws.append(Wo)
            return fesc,ProbSets,Ws
        
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
            Muv,Beta=physParams[0],physParams[1]
            cMuv,cBeta,cte= mathParams[0],mathParams[1],mathParams[2]
            #print(((Muv+20)*cMuv) ,((Beta+2)*cBeta)   ,   cte )
            Parameter   =   ((Muv+20)*cMuv) + ((Beta+2)*cBeta)   +   cte 
            return Parameter

        def OriginalLikelihood(self,ew,A,Wo):
            print(Wo)
            return ((A/Wo)*np.exp(-ew/Wo)*np.heaviside(ew,0.0) + (1-A)*DeltaFunc(ew) )
        
        def OriginalLikelihood2(self,ew,Wo):
            return ((1/Wo)*np.exp(-ew/Wo)*np.heaviside(ew,0.0))

        def RandomDrawEW(self,noise): 
            #print(self.EW[0:5])
            return [np.random.normal(loc=w,scale=noise) for w in self.fesc]

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

        def Classify_fesc(self):
            size = len(self.fesc_obs)

            types = np.random.choice(["LyaHa","Ha","Lya"], size=size)

            self.types = types
            return types
        
        def DetectionTypeReal(self):
            self.types=[]
            for i in range(0,len(self.fesc_obs)):
                if self.typesCat[i]=="LAE" and np.isnan(self.dfesc[i])==False:
                    self.types.append("LyaHa")
                elif self.typesCat[i]=="LAE" and np.isnan(self.dfesc[i])==True:
                    self.types.append("Lya")
                elif self.typesCat[i]=="NonDetect" and np.isnan(self.dfesc[i])==True:
                    self.types.append("Ha")
            self.types=np.array(self.types)
            return  self.types  
        
        def GenerateWtabUp(self,fesc_up):
            start_value = -0.5
            step = 0.01
            end_value=fesc_up
            num_elements = int((end_value - start_value) / step) + 1
            self.wtab=np.round(np.linspace(start_value, end_value, num_elements),1)
            return np.round(np.linspace(start_value, end_value, num_elements),1)
        
        def GenerateWtabLow(self,fesc_low):
            start_value = fesc_low
            step = 0.01
            end_value=1.5
            num_elements = int((end_value - start_value) / step) + 1
            self.wtab=np.round(np.linspace(start_value, end_value, num_elements),1)
            return np.round(np.linspace(start_value, end_value, num_elements),1)
        
        def GenerateWtab_real(self):
            wtabs=[]
            for tipo,fesc in zip(self.types,self.fesc_obs):

                if tipo=="Ha":

                    start_value = -0.5
                    step = 0.05
                    end_value=fesc
                    num_elements = int((end_value - start_value) / step) + 1
                    wtabs.append(np.round(np.linspace(start_value, end_value, num_elements),3))

                elif tipo=="Lya":
                    start_value = fesc
                    step = 0.05
                    end_value=3.
                    num_elements = int((end_value - start_value) / step) + 1
                    wtabs.append(np.round(np.linspace(start_value, end_value, num_elements),3))
                
                else:
                    wtabs.append(["Not Needed"])
                
            self.wtab=np.array(wtabs)
            return self.wtab

