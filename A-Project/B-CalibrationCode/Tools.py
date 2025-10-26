import numpy as np
import glob
from astropy.io import fits
from Spectra import *
from SpectraData import *
from Inventory import *
from scipy.optimize import minimize
from HighSeeing import HighSeeing_Filter
import pandas





############
#Equations and transformations
############
def angstromTokms(centralA,As):
    c=300000
    v=c*(As-centralA)/centralA
    return v+c

def kmsToangstrom(v,centralA):
    c=300000
    As = centralA*(v+c)/c
    return np.abs(centralA-As)

def redshiftToWave(z):
    wave=1215.67*(1+z)
    return wave

def WaveToRedshift(wave):
    z=(wave/1215.67)-1
    return z

#####
#Data management
#####


def GetLAEs(df):
    """"
    Reads the catalogs with the wavelength and Ypos
    that I eye searched before

    df : csv file
    
    returns:
    coords: Y position and wavelegnth of emission line
    """
    mascara=[]
    for name in df["name"].values:
        if "HST" not in name and "f_" not in name:
            mascara.append(True)
        else: 
            mascara.append(False)
    df=df[mascara]
    coords=[]
    for i in df["coord"].values:
        if "," in i:
            d=i.split(",")
        elif "_" in i:
            d=i.split("_")
        else:
            d="X"
        try:
            if float(d[0])/1000.0>1:
                coords.append([float(d[1]),float(d[0])])
            elif float(d[0])/1000.0<1:
                coords.append([float(d[0]),float(d[1])])

        except:
            coords.append([np.nan,np.nan])


    return coords

def nonDetections(df,catalog):
    """"
    Reads the catalogs and gets whatever doesnt have a lyman detection (only Include possible LAES)
    normally should be an X or other type of text

    df : csv file
    
    returns:
    Ids of non detections
    """
    mascara=[]
    for name in df["name"].values:
        if "HST" not in name and "f_" not in name:
            mascara.append(True)
        else: 
            mascara.append(False)
    df=df[mascara]
    Mask=[]
    names=df["name"].values
    for n in names:
        for c in catalog:
            if n==c["id_charlotte"]:
                found=c

#            print(found["Type"])
#        if found["Type"]=="LAE":
#            Mask.append(False)
#        if found["Type"]=="NonDetect":
#            Mask.append(True)

    return Mask,names

def nonDetections_everything(df):
    """"
    Reads the catalogs and gets whatever doesnt have a lyman detection (Includes everything)
    normally should be an X or other type of text

    df : csv file
    
    returns:
    Ids of non detections
    """
    mascara=[]
    for name in df["name"].values:
        if True==True:
            mascara.append(True)
        else: 
            mascara.append(False)
    df=df[mascara]
    Mask=[]
    for i,n in zip(df["coord"].values,df["name"].values):
        i=str(i)
        if i=="nan":
            Mask.append(False)
        elif "," in i:
            Mask.append(False)
        elif "_" in i:
            Mask.append(False)
        else:
            Mask.append(True)

    return Mask

def readDataLBG(location):
    #data_dir = '../data/'
    fname_data = np.sort(glob.glob(location)) #read data
    fname_data = [i for i in fname_data if "3DHST" not in i and "f_" not in i] #takes only LAEs
    Specs=[]
    for data in fname_data[:]:
        HDU = fits.open(data)
        flux=HDU[0]
        error=HDU[1]
        Inventory(Spectra(flux,SpectraData(flux,error))).addSpectra(Specs,flux)  #This is the class that has many functions to manipulate the data
    return np.array(Specs)

def readALLData(location):
    data_dir = 'data/'
    fname_data = np.sort(glob.glob(data_dir+location)) #read data
    Specs=[]
    for data in fname_data[:]:
        HDU = fits.open(data)
        flux=HDU[0]
        error=HDU[1]
        Inventory(Spectra(flux,SpectraData(flux,error))).addSpectra(Specs,flux)  #This is the class that has many functions to manipulate the data
    return np.array(Specs)

def gaussian_ML(x,flux, mu, sig,C) :
    """Gaussian"""
    return flux* np.exp( -0.5 * (x-mu)**2 / sig**2) + C

def LogNormal_ML(x,flux, mu, sig,C) :
    """log Normal"""
    return (flux/x)* np.exp( -0.5 * (np.log(x)-mu)**2 / sig**2) + C

def log_likelihood_NormalML(theta, x, y, yerr):
    flux, mu,sig,C = theta
    model = LogNormal_ML(x,flux, mu,sig,C)
    sigma2 = yerr ** 2  #sigma2 is sigma square from the likelihood stuff, not the gaussian
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_likelihood_ML(theta, x, y, yerr):
    flux, mu,sig,C = theta
    model = gaussian_ML(x,flux, mu,sig,C)
    sigma2 = yerr ** 2  #sigma2 is sigma square from the likelihood stuff, not the gaussian
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def FitGaussian(X,Y,Yerr,inFlux,inMu,inSTD,inC):
    nll = lambda *args: -log_likelihood_ML(*args)   #ML fit
    initial = np.array([inFlux,inMu,inSTD,inC ])
    soln = minimize(nll, initial, args=(X,Y,Yerr))
    flux,mu,sig,C = soln.x
    return flux,mu,sig,C


def Obtain_As(fname_data,fname_err,mask_name,amount=None,skyscale=None):
    ALL_A=[]
    Which={}
    Pick="LBG"
    Which["Filler"]="3DHST"
    Which["LBG"]="3DHST" #just put the same and do "not in"
    Which["Any"]="" #Check if this works
    MasknonDetect    =   nonDetections(pandas.read_csv('../Docs/'+mask_name+'.csv',sep=";"))[0]

    contar=0
    for data,err in zip(fname_data[:amount],fname_err[:amount]):
        contar+=1
        print(contar, data)
        if HighSeeing_Filter(data,mask_name) == False:
            print(data,": Skipped")
            continue

        hdu_list_flux = fits.open(data)
        hdu_list_err = fits.open(err)

        Specs=[]

        for i in range(0,len(hdu_list_flux)): # only for a couple, not len(hdu_list_flux)
            if Pick=="LBG":
                if "ImageHDU" in str(hdu_list_flux[i]) and Which[Pick] not in hdu_list_flux[i].header["SLITOBJ"] and "f_" not in hdu_list_flux[i].header["SLITOBJ"]: # WE ONLY GRAB ImageHDU 's
                    flux=hdu_list_flux[i]
                    error=hdu_list_err[i]
                    Inventory(Spectra(flux,SpectraData(flux,error))).addSpectra(Specs,flux) #Spectra object needs ID,RA,DEC,Z & SpectraData object where we put our spectra

            

        Skies=[]
        for s in np.array(Specs)[MasknonDetect]:
            Skies.append(s.Data.skySpectra())

        SSpectra=np.nanmedian(Skies,axis=0)

        As=[]
        for s in Specs:
            A=s.Data.processSlit(SSpectra,skyscale=None)
            As.append(A)
        ALL_A.append(As)
    return ALL_A




#########
#SCALING NOISE
#########
def SideMask(NonDetectSpec):
    SideA,SideB=[],[]
    for s in NonDetectSpec:
        if s.Data.header["SIDE"]=="A":
            SideA.append(True)
            SideB.append(False)
        elif s.Data.header["SIDE"]=="B":
            SideA.append(False)
            SideB.append(True)
    return SideA,SideB

def StoN_calculation(Spec,wave):
    Mus,Sigs=[],[]
    for s in Spec:
        Wavemask= ( (wave > 7638.) & (wave < 7707.) ) | ( (wave > 7893.) & (wave < 7908.) ) | ( (wave > 8106.) & (wave < 8271.) ) \
        | ( (wave > 8314.) & (wave < 8337.) ) | ( (wave > 8469.) & (wave < 8489.) ) | ( (wave > 8508.) & (wave < 8534.) ) \
        | ( (wave > 8552.) & (wave < 8754.) ) | ( (wave > 8795.) & (wave < 8821.) ) | ( (wave > 9006.) & (wave < 9300.) ) \
        | ( (wave > 9571.) & (wave < 9603.) ) | ( (wave > 9625.) & (wave < 9665.) ) | ( (wave > 9749.) & (wave < 9785.) )  
        #Wavemask=(wave > 9860.) & (wave < 9984.)
        EmptyWaves =    wave[Wavemask]
        EmptyData   = s.Data.rawData[3:-3,Wavemask]  
        EmptyErr   = s.Data.error[3:-3,Wavemask]  
        value,b,_=plt.hist(EmptyData.flatten()[EmptyData.flatten()!=0.0]/EmptyErr.flatten()[EmptyData.flatten()!=0.0],bins=80,range=[-5,5])
        # Divide the values by the standard deviation i get
        plt.clf()
        bins=[(b[i]+b[i+1])/2 for i in range(0,len(b)-1)]
        #plt.step(bins,value/max(value),where="mid")
        valueErrs=np.ones(len(value))*0.01
        vals=FitGaussian(bins,value/max(value),valueErrs,0.1,0,0.5,0)
        #plt.plot(bins,gaussian_ML(bins,*vals))
        Mus.append(vals[1])
        Sigs.append(vals[2])
        #plt.show()
    Mus=np.array(Mus)
    Sigs=np.array(Sigs)
    return Mus,Sigs

def ScaleNoise(maskName,version=None):
    MasknonDetect    =   nonDetections(pandas.read_csv('../Docs/'+maskName+'.csv',sep=";"))[0]
    Spec = readDataLBG("large_files/Reduced_Data/"+maskName+"/2D/*.fits")
    if version=="noiseCorrected":
        Spec = readDataLBG("large_files/Reduced_Data/"+maskName+"/2D/noiseCorrected/*.fits")     
    NonDetectSpec   =   Spec[MasknonDetect]
    wave=np.arange(6760.0,6760.0+5631*0.620000004768,0.620000004768)
    SideA,SideB=SideMask(NonDetectSpec)
    Mus,Sigs=StoN_calculation(NonDetectSpec,wave)
    Sides={}
    Sides["A"]=np.median(np.abs(Sigs[SideA]))
    Sides["B"]=np.median(np.abs(Sigs[SideB]))
    return Sides

"""
def GetStrongSky(dataFits,ShowPlots=False):
    wave=np.arange(6760.0,6760.0+5631*0.620000004768,0.620000004768)
    hdu=fits.open(dataFits)
    noise=np.nansum(hdu[0].data,axis=0)
    wave=wave[noise>np.nanmedian(noise)-0.5*np.nanstd(noise)]
    noise=noise[noise>np.nanmedian(noise)-0.5*np.nanstd(noise)]

    StrongSky=[]
    for i in range(0,len(wave)):
        if noise[i]>=(np.nanmedian(noise)+2*np.nanstd(noise)):
            StrongSky.append(wave[i])

    print(StrongSky)
    newSpec=[]
    for i in range(0,len(wave)):
        if wave[i] not in StrongSky:
            newSpec
            newSpec.append(noise[i])
    if ShowPlots==True:
        plt.plot(noise)
        plt.axhline((np.nanmedian(noise)))
        plt.axhline((np.nanmedian(noise)+2*np.nanstd(noise)))
        plt.plot(newSpec)
        plt.ylabel("Flux")
        plt.xlabel("Wavelength [A]")
        plt.title("Code to find the Strong sky lines")
        plt.show()

    return StrongSky
"""


def GetStrongSky(file="/Users/gonzalo/Desktop/Code/Gonzalo_Binospec/A-Project/A-Catalogs/Telluric/gident_860L.dat"):
    catSky=ascii.read(file)
    lines=catSky[catSky["int"]>200]
    return list(lines["wave"])