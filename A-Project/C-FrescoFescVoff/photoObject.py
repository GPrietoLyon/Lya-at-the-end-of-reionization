import Tools
import numpy as np
import MCMC
import matplotlib.pyplot as plt
import random

class PhotoObject:
    def __init__(self,catWave,Other,Extra=[None]):
        """
        In this class I will manipulate all the photometry and get things from it for each source
        
        Args:
            catalog: catalog with binospec coords and Candels photometry
        """
        self.ID          = catWave["ID"]
        self.IDFresco          = Other[0]
        self.zLya= Other[1]
        self.zLyaDist=Extra
        self.aWaveHa = catWave["Ha"]
        self.aWaveOiii = catWave["Oiii"]    
        self.WaveLines,self.fwhmLines=None,None  
        self.off = None     
        self.fwhm = None   
        self.Muv=None
        self.zsys=None
        self.ra=Other[2]
        self.dec=Other[3]
        #self.cat=catG
        #self.museID     = museid
        #self.d2d        =D2D
        #self.Phot   =   photcat
        #self.zLyaPeak=zLya
        #self.zLine    =None
        #self.off    =None  
        #self.fwhm               =None
        #self.Means=Means
        #self.FWHMs=FWHMs
        #self.Muv_ch,self.dMuv_ch=MUVs["MUV"],MUVs["MUVerr"]
        #self.Muv=None
        #self.Slope=None
        #self.mu,self.dmu=magnif["MAGNIFICATION_50"],((magnif["MAGNIFICATION_50"]-magnif["MAGNIFICATION_16"])+(magnif["MAGNIFICATION_84"]-magnif["MAGNIFICATION_50"]))/2.0
        #self.EW,self.dEW=np.abs(EW[2][0]),EW[2][1]
        #self.dataEW=EW[0]
        #self.errorsEW=EW[1]
        #self.LyaCont,self.dLyaCont=None,None
        #self.LyaZErr_ATLAS=dKMS
        #self.Flya=Flya
        #self.GetFWHM()
        #self.GetRedshift()
        #self.GetOffset()
        #self.EWerr()
    def WhichLine(self,lineName):
        self.LineName=lineName

    def getSpectra(self,wave,flux,err):
        self.wave=wave
        self.flux=flux
        self.err=err

    def giveWave(self,Waves):
        self.WaveLines=Waves

    def giveParam(self,Param,Muv):
        self.Muv=Muv

    def giveFWHM(self,FWHM):
        self.fwhmLines=FWHM
    
    def giveAmplitude(self,amp):
        self.amplitudeLines=amp

    def giveCte(self,C):
        self.CteLines=C

    def givePairs(self,pairs):
        self.pairsLines=pairs

    def givezSysDist(self,pairs):
        self.zSysDist=pairs

    def getOffset(self):
        lines=["Ha","Oiii"]
        off={"Ha":None,"Oiii":None} 
        for l in lines:
            zsys= Tools.LineRedshift(l,self.WaveLines[l])
            if np.isnan(zsys[0])==False:
                self.zsys=zsys
            zlya=self.zLya
            off[l]=Tools.vOffset(zlya,zsys)
        #self.off=off
        self.off=off[self.LineName]#Tools.mergeLineData(off["Ha"],off["Oiii"])
        val=self.off
        #print(val)
        self.doff=np.mean([val[1]-val[0],val[2]-val[1]])
        self.off=val[1]

    def getOffsetLyaError(self):
        lines=["Ha","Oiii"]
        off={"Ha":None,"Oiii":None} 
        zlya=np.array(self.zLyaDist)
        for l in lines:
            print("AAAAAA")
            zsys= Tools.LineRedshift(l,self.WaveLines[l+"Dist"])
            zsys=np.array(random.choices(zsys,k=len(zlya)))
            if np.isnan(zsys[0])==False:
                self.zsys=[np.median(zsys)-np.std(zsys),np.median(zsys),np.median(zsys)+np.std(zsys)]
                print(self.zsys)

            off[l]=Tools.vOffset(zlya,zsys)

        self.off=np.median(off[self.LineName])
        #print(val)
        self.doff=np.std(off[self.LineName])
        print(self.off,self.doff)

    def getFWHM(self):
        lines=["Ha","Oiii"]
        fwhm={"Ha":None,"Oiii":None} 
        for l in lines:
            print(self.fwhmLines)
            Ang=self.fwhmLines[l]
            mean=self.WaveLines[l][1]
            #print(Ang,mean)
            fwhm[l]=Tools.angstromTokms(mean,Ang/2.0)*2
        self.fwhm=fwhm["Ha"]#Tools.mergeLineData(fwhm["Ha"],fwhm["Oiii"])
        val=self.fwhm
        self.dfwhm=np.mean([val[1]-val[0],val[2]-val[1]])
        self.fwhm=val[1]

    def getFlux(self):
        lines=["Ha","Oiii"]
        flux={"Ha":None,"Oiii":None} 
        for l in lines:
            f=self.amplitudeLines[l]*1e-18
            flux[l]=f
        self.flux=flux["Ha"]#Tools.mergeLineData(fwhm["Ha"],fwhm["Oiii"])
        val=self.flux
        self.dflux=np.mean([val[1]-val[0],val[2]-val[1]])
        self.flux=val[1]


    """
    def giveMUV(self,muv):
        self.Muv=muv
    def giveSlope(self,slope):
        self.Slope=slope

    def EWerr(self):
        if self.EW>100000:
            lya,uv=self.dataEW[0],self.dataEW[1]
            self.EW=lya/uv
            self.dEW=np.nan

    def giveLyaCont(self,cont):
        self.LyaCont=cont[1]
        self.dLyaCont=((cont[1]-cont[0])+(cont[2]-cont[0]))/2.0

    def GetEW(self):
        continuumWave=1241.5*(1+self.zLya)
        lyaflux=self.FLya

        UVflux    =   (self.LyaCont)*(2.998e18/(continuumWave**2))
        EW      =   (lyaflux)/UVflux
        EWrest=EW/(1+self.zLya)
        print("UV:",UVflux)
        print("Lya :",lyaflux)
        print(EW,EWrest)


    def GetFWHM(self):
        lines=["Ha","OIII"]
        fwhm={"Ha":None,"OIII":None} 
        for l in lines:
            Ang=self.FWHMs[l]
            mean=self.Means[l][1]
            #print(Ang,mean)
            fwhm[l]=Tools.angstromTokms(mean,Ang/2.0)*2
        self.fwhm=fwhm

    def GetRedshift(self):
        lines=["Ha","OIII"]
        z={"Ha":None,"OIII":None} 
        for l in lines:
            wave=self.Means[l]
            z[l]=Tools.LineRedshift(l,wave)
        self.zLine=z

    def GetOffset(self):
        lines=["Ha","OIII"]
        off={"Ha":None,"OIII":None} 
        for l in lines:
            zsys= self.zLine[l]
            zlya=self.zLya
            off[l]=Tools.vOffset(zlya,zsys)
        self.off=off
    """

 