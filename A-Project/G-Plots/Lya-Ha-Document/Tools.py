from astropy.io import ascii
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.optimize import minimize
import math
import scipy.special as scispe

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def WhereHa(zspec):
    return 0.65628*(1+zspec)

def WhereOiii(zspec):
    return 0.50082*(1+zspec)

def isLine(line,lmin,lmax):
    if lmin<line<lmax:
        return True
    else:
        return False

def angstromTokms(centralA,As):
    c=299792
    v=c*(As-centralA)/centralA
    return v+c

    
def kmsToangstrom(v,centralA):
    c=299792
    As = centralA*(v+c)/c
    return np.abs(centralA-As)

def kmsToangstrom2(v,centralA):
    c=299792
    As = centralA*(v+c)/c
    return centralA-As

def redshiftToWave(z):
    wave=1215.67*(1+z)
    return wave

def WaveToRedshiftAir(w):
    z=(w/1215.32)-1
    return z

def LineRedshift(l,wave):
    rest={"Ha":6564.614,"Oiii":5008.239}
    z=(wave/rest[l])-1
    return z

def vOffset(zLya,zSys):
    c=299792.458 #km/s
    offset=c*((zLya-zSys)/(1+zSys))
    return offset

def mergeLineData(Ha,Oiii):
    Ha=np.transpose(Ha)
    Oiii=np.transpose(Oiii)
    if np.isnan(Ha[0]):
        v1=Oiii[2]
        vm=Oiii[1]
        vp=Oiii[0]
    if np.isnan(Ha[0])==False:    
        v1=Ha[2]
        vm=Ha[1]
        vp=Ha[0]       
    return np.array([v1,vm,vp])

def ReadCatalogs(name):
    '''
    Helps to read the catalogs

    Input:

    - name : location of the catalog

    Output:

    - df : Catalog in dictionary/ascii.astropy format
    
    
    '''
    df = ascii.read(name)
    return df

def MakeNewCat(bino,candels,props,ObservedIDs,photoz,Flim,MaskLAE):
    '''
    Makes a joint catalog between the Binospec cats and the Candels catalog

    Input:

    - bino : catalog from binospec
    - candels : catalog from candels
    - props : any Lya properties we might want to add
    - ObservedIDs: There are some not-observed objects in bino. This list has all that we did observe (we didnt detect them all tho)
    * Both must be in the format given by ReadCatalogs()
    
    Variables:

    - Filter names and FiltErrs, are the names of filters in the catalog. If different catalog labels or different filters are needed, this must be changed
    
    Output:

    -No output, but writes file to folder with final catalog 
    '''
    #Mask bino catalog for the things that we did observe

    m=[True if x in ObservedIDs else False for x in list(bino["id_charlotte"])]
    bino=bino[m]
    # Get a mix of both catalogs
    ID_MATCHES=bino["id_candels"]
    CandelsLBG=[]
    for ID in ID_MATCHES[:]:
        for gal in candels:
            if gal["ID"] ==ID:
                mags=Change2Mag(gal)
                CandelsLBG.append(mags)

    filterNames     =   ['KPNO_U_FLUX', 'LBC_U_FLUX', 'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 'ACS_F775W_FLUX', 'ACS_F814W_FLUX', 'ACS_F850LP_FLUX', 'WFC3_F105W_FLUX', 'WFC3_F125W_FLUX', 'WFC3_F140W_FLUX', 'WFC3_F160W_FLUX']
    filtErrNames    =   [ 'KPNO_U_FLUXERR', 'LBC_U_FLUXERR', 'ACS_F435W_FLUXERR', 'ACS_F606W_FLUXERR', 'ACS_F775W_FLUXERR', 'ACS_F814W_FLUXERR', 'ACS_F850LP_FLUXERR', 'WFC3_F105W_FLUXERR', 'WFC3_F125W_FLUXERR', 'WFC3_F140W_FLUXERR', 'WFC3_F160W_FLUXERR']
    columns=list(bino.columns)+list(filterNames+filtErrNames)+list(props.columns)[1:]+list(["photoz","IDFink"])+list(["FluxLim"]+list(["Type"]))
    CATALOG=[]
    for i in range(0,len(bino)):
        CATALOG.append(np.array(list(bino[i])+list(CandelsLBG[i])))
        
    propCols=list(props.columns)[1:]
    Properties=[]
    for i in CATALOG:
        zFound=0
        for j in props:
            if str(i[1])==str(j[0]):
                zFound=1
                Properties.append(j[propCols])

        if zFound==0:
            Properties.append([np.nan for i in range(0,len(propCols))])
    
    CATALOG=[]
    for i in range(0,len(bino)):
        data=list(bino[i])+list(CandelsLBG[i])+list(Properties[i])
        #Add photoz
        found=0
        for j in photoz[0]:
            if bino[i][1] == j["ID"] and found==0:
                found=1
                data=data+[float(j["z_Fink"])]+[j["ID_Fink"]]

        for j in photoz[1]:
            if bino[i][1] == j["ID"] and found==0:
                found=1
                data=data+[float(j["z_fink"])]+["nan"]
        CATALOG.append(np.array(data))

    CATALOG2=[]
    for gal in CATALOG:
        fluxLimit=Flim.item().get(gal[1])
        ngal=list(gal)
        if fluxLimit==None:
            ngal.append(np.nan)
        else:
            ngal.append(fluxLimit)
        CATALOG2.append(ngal)
    
    CATALOG=[]
    for i in range(0,len(CATALOG2)):
        gal=list(CATALOG2[i])
        if MaskLAE[i]==True:
            gal.append("LAE")
        if MaskLAE[i]==False:
            gal.append("NonDetect")
        CATALOG.append(gal)


    
    ascii.write(np.array(CATALOG),"Catalogs/Binospec-Candels.cat",names=np.array(columns),overwrite=True)

def Flux2Mag(f):
    '''
    Equation that transforms flux to ABMag

    Input:

    - f: flux in micro Janskys

    Output:

    - AB Magnitude
    
    '''
    return -2.5*np.log10(f*1e-6)+8.9

def Change2Mag(photometry):

    '''
    Takes fluxes and gives magnitudes.
    Also keeps -99.0 non detections as -99.0

    Input:

    - photometry: Catalog with format from ReadCatalogs()

    Variables:

    - Filter names and FiltErrs, are the names of filters in the catalog. If different catalog labels or different filters are needed, this must be changed

    Output:

    - return : List with all Magnitudes followed by all errors of magnitudes
    
    '''

    filterNames     =   ['KPNO_U_FLUX', 'LBC_U_FLUX', 'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 'ACS_F775W_FLUX', 'ACS_F814W_FLUX', 'ACS_F850LP_FLUX', 'WFC3_F105W_FLUX', 'WFC3_F125W_FLUX', 'WFC3_F140W_FLUX', 'WFC3_F160W_FLUX']
    filtErrNames    =   [ 'KPNO_U_FLUXERR', 'LBC_U_FLUXERR', 'ACS_F435W_FLUXERR', 'ACS_F606W_FLUXERR', 'ACS_F775W_FLUXERR', 'ACS_F814W_FLUXERR', 'ACS_F850LP_FLUXERR', 'WFC3_F105W_FLUXERR', 'WFC3_F125W_FLUXERR', 'WFC3_F140W_FLUXERR', 'WFC3_F160W_FLUXERR']

    Flux            =   photometry[filterNames]
    Error           =   photometry[filtErrNames]

    Mag={}
    MagErr={}
    for i in range(0,len(Flux)): 
        if Flux[i]==-99.0 and Error[i]==-99.0:
            Mag[filterNames[i]]=-99.0
            MagErr[filterNames[i]]=-99.0      

        else:
            m=Flux2Mag(Flux[i])
            merr=Flux2Mag(np.abs(Flux[i]+Error[i]))
            Mag[filterNames[i]]=m
            MagErr[filterNames[i]]=np.abs(m-merr)

    return list(Mag.values())   +   list(MagErr.values())

def removeBands(Cwaves,EffW,z,ShowPlots=False):
    '''
    Sacar bandas que caen a la izquierde de lyman alpha
    '''
    BluerThanLyaMask=[]
    Cwaves  =   np.array(Cwaves)
    EffW    =   np.array(EffW)   
    Limit   =   1215.6 * (z+1)
    for Wave,Width in zip(Cwaves,EffW):
        if Wave-Width/2 < Limit:
            BluerThanLyaMask.append(False)
        if  Wave-Width/2 >= Limit:
            BluerThanLyaMask.append(True)

    return BluerThanLyaMask

def PowerLaw(x, beta, c):
    """
    Not really a power law since we are at log scale
    """
    return x*beta +c 


def FitPowerLaw(photometry,filters,Cwaves,ShowPlots=True):
    Cwaves=np.array(Cwaves)
    Mags    = np.array(list(photometry[list(filters[:int(len(filters)/2)])]))
    MagsErr = np.array(list(photometry[filters[int(len(filters)/2):]]))
    #print(Mags,list(filters[:int(len(filters)/2)]) )

    maskNinetyNines = Mags!=-99.0
    Mags,MagsErr,Cwaves    =   Mags[maskNinetyNines],MagsErr[maskNinetyNines],Cwaves[maskNinetyNines]
   
    fluxes=10**(-(Mags+48.6)/2.5)
    #print(Mags,MagsErr)
    flxMinErr=10**(-(Mags+MagsErr+48.6)/2.5)
    flxPlusErr=10**(-(Mags-MagsErr+48.6)/2.5)

    fluxesErr=np.median([flxPlusErr-fluxes,fluxes-flxMinErr],axis=0)

    #print(fluxes,flxPlusErr,flxMinErr,fluxesErr)
    logFerr=np.median([np.log10(fluxes-fluxesErr)-np.log10(fluxes),np.log10(fluxes)-np.log10(fluxes+fluxesErr)],axis=0)
    logFerr[np.isnan(logFerr)==True]=0.01

    


    #fluxes=fluxes*(2.998e18/(Cwaves**2)) #convert to f lambda

    #popt, pcov = curve_fit(PowerLaw, np.log10(Cwaves), np.log10(fluxes),absolute_sigma=True,sigma=logFerr,p0=[1,-30])
    try:
        popt,pcov=np.polyfit(np.log10(Cwaves),np.log10(fluxes),deg=1,w=1/logFerr,cov=True)
    except:
        popt=np.polyfit(np.log10(Cwaves),np.log10(fluxes),deg=1,w=1/logFerr)
        pcov=[[0.1,0.1],[0.1,0.5]]

    #print("BetaSlope = ",popt[0]-2 )
    
    if ShowPlots    ==  True:
        plt.errorbar(np.log10(Cwaves),np.log10(fluxes),yerr=logFerr,fmt=".")
        plt.plot(np.log10(Cwaves),PowerLaw(np.log10(Cwaves),*popt))
        plt.gca().invert_yaxis()
        plt.title("beta: "+ str(popt[0]-2))
        plt.ylabel("log Flux")
        plt.xlabel("log(Wavelength)")
        plt.show()


    return popt[0],popt[1],[np.sqrt(np.diag(pcov))],[Mags,MagsErr]

def GetMuv(fitParams,z):
    cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
    b,c     =   fitParams[0],fitParams[1]
    logflux =   c+b*np.log10(1500*(1+z))
    flux    =   (10**logflux)/(1+z)
    m       =   -2.5*np.log10(flux) -48.6
    dL      =   cosmo.luminosity_distance(z).value
    Mabs    =   m-5*(np.log10(dL*10**6) - 1)
    return Mabs

def GetLuminosity(flux,z):
    flux    =   10**flux
    cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
    dL      =   cosmo.luminosity_distance(z).cgs
    Lum     =   flux*(u.erg*u.cm**(-2)*u.s**(-1))*4.0*np.pi*dL**2
    return Lum.value

def MaskData(data,mask):
    masked=[]
    for i in data:
        masked.append(i[mask])
    return masked

def GetEW(Lyaflux,fitParams,z):
    b,c     =   fitParams[0],fitParams[1]
    continuumWave=1241.5*(1+z)
    logflux =   c+b*np.log10(continuumWave)
    UVflux    =   (10**logflux)*(2.998e18/(continuumWave**2))
    EW      =   (10**Lyaflux)/UVflux
    print(10**Lyaflux,UVflux,EW)
    EWrest=EW/(1+z)
    return EWrest

########

def DeltaDirac(ew):
    ew=np.array(ew)
    mask= ew==0
    ew[mask]=1
    ew[~mask]=0
    return ew



##### MCMC


def skewed_gaussian(x,flux, mu, FWHM,g,C) :
    """Gaussian"""
    d=np.power(10,g)/(np.sqrt(1+(np.power(10,g))**2))
    fw_param=(FWHM/(2*np.sqrt(2*np.log(2)))/np.sqrt((1-(2*d**2/np.pi))))
    return (flux/(fw_param*np.sqrt(2*np.pi))) * np.exp( -0.5 * (x-mu)**2 / fw_param**2)* (1+scispe.erf( (10**g)*(x-mu)/(fw_param*np.sqrt(2)))) + C


def log_prior(theta,meanL,meanG):
    flux, mu,FWHM,g,C = theta
    if 0 < flux < 10. and 0. < FWHM < 30. and np.log10(0.1) < g < np.log10(15)  and -1 < C < 1: #way to get the ranges and define as finite when they are inside the ranges
        return 0.0 
    return -np.inf

def log_likelihood(theta, x, y, yerr):
    flux, mu,FWHM,g,C = theta
    model = skewed_gaussian(x,flux, mu,FWHM,g,C)
    sigma2 = yerr ** 2  #sigma2 is sigma square from the likelihood stuff, not the gaussian
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, x, y, yerr,meanL,meanG):
    lp = log_prior(theta,meanL,meanG)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


##### ML


def skewed_gaussian_ML(x,flux, mu, sig,g,C) :
    """Gaussian"""

    return (flux/(sig*np.sqrt(2*np.pi))) * np.exp( -0.5 * (x-mu)**2 / sig**2)* (1+scispe.erf( g*(x-mu)/(sig*np.sqrt(2)))) + C

def skewed_gaussian_modif(x,flux, mu, sig,g,C,rmin,rmax) :
    """Gaussian"""
    result= (flux/(sig*np.sqrt(2*np.pi))) * np.exp( -0.5 * (x-mu)**2 / sig**2)* (1+scispe.erf( g*(x-mu)/(sig*np.sqrt(2)))) + C
    mask = (x < rmin) | (x > rmax)
    result[mask]=0
    return result


def log_prior_ML(theta,meanL,meanG):
    flux, mu,FWHM,g,C = theta
    if 0 < flux < 10. and 0. < FWHM < 30. and 0 < g < 15  and -1 < C < 1: #way to get the ranges and define as finite when they are inside the ranges
        return 0.0 
    return -np.inf



def log_likelihood_ML(theta, x, y, yerr):
    flux, mu,FWHM,g,C = theta
    model = skewed_gaussian_ML(x,flux, mu,FWHM,g,C)
    sigma2 = yerr ** 2  #sigma2 is sigma square from the likelihood stuff, not the gaussian
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability_ML(theta, x, y, yerr,meanL,meanG):
    lp = log_prior_ML(theta,meanL,meanG)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ML(theta, x, y, yerr)

#sigma to FWHM, amp to flux and g to skewness








#####
#ChatGPT stuff
#####

def plot_points_with_numbers(points):
    # Extract x and y coordinates from the points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Create a scatter plot of the points
    plt.scatter(x_coords, y_coords)

    # Add numbers to each point
    for i, point in enumerate(points):
        plt.text(point[0], point[1], str(i+1), fontsize=12, verticalalignment='bottom')

    # Set labels 


def convert_string_to_2d_array(s):
    # Remove parentheses and split the string into individual coordinates
    coordinates = s.strip('()').split('),(')

    # Initialize an empty 2x4 array
    array_2d = [[0.0] * 4 for _ in range(2)]

    # Iterate over each coordinate and convert to float
    for i, coordinate in enumerate(coordinates):
        # Split the coordinate into latitude and longitude
        lat, lng = coordinate.split(',')

        # Convert latitude and longitude to float
        lat_float = float(lat)
        lng_float = float(lng)

        # Assign values to the 2D array
        array_2d[0][i] = lat_float
        array_2d[1][i] = lng_float

    return array_2d

def find_coordinate_between_points(X1, Y1, X2, Y2):
    # Calculate the average X-coordinate
    avg_X = (X1 + X2) / 2

    # Calculate the average Y-coordinate
    avg_Y = (Y1 + Y2) / 2

    return avg_X, avg_Y


def point_on_side_of_line(x1, y1, x2, y2, x3, y3):
    # Calculate the slope
    slope = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept
    y_intercept = y1 - slope * x1

    # Calculate the value using the line equation
    calculated_value = slope * x3 + y_intercept

    # Compare the value with the y-coordinate of the third point
    if calculated_value > y3:
        return True  # The point is on one side of the line
    else:
        return False  # The point is on the other side of the line
