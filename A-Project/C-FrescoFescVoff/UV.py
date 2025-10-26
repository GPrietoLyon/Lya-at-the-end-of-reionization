import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import MCMCUV
import emcee
import corner
from IPython.display import display, Math
from astropy.table import Table

def removeBands(Cwaves,EffW,z,cat,filter_names,ShowPlots=False):
    '''
    Sacar bandas que caen a la izquierde de lyman alpha y a la derecha del break
    '''
    Mask=[]
    filter_names=filter_names[:int(len(filter_names)/2)]
    Cwaves  =   np.array(Cwaves)
    EffW    =   np.array(EffW)   
    LyaLimit   =   1250 * (z+1)
    BreakLimit  =   2600*(z+1)

    for Wave,Width in zip(Cwaves,EffW):
        if  Width[0] > LyaLimit and Width[1] < BreakLimit:
            #print("WIDE",LyaLimit,Width[0],Width[1],BreakLimit,"PASS")
            Mask.append(True)
        else:
            #print("WIDE",LyaLimit,Width[0],Width[1],BreakLimit,"No PASS")
            Mask.append(False)



    for i, name in enumerate(filter_names):
        if np.isnan(cat[name]) or cat[name] < 0:
            Mask[i] = False


    
    if np.sum(Mask) < 3:
        LyaLimit = 1216 * (z + 1)
        BreakLimit = 2800 * (z + 1)
        Mask = []
        for Wave, Width,name in zip(Cwaves, EffW,filter_names):
            if Width[0] > LyaLimit and Width[1] < BreakLimit and (np.isnan(cat[name])==False or cat[name] > 0):
                Mask.append(True)
            else:
                Mask.append(False)


    if np.sum(Mask) < 3:
        LyaLimit = 1216 * (z + 1)
        BreakLimit = 3000 * (z + 1)
        Mask = []
        for Wave, Width in zip(Cwaves, EffW):
            if Width[0] > LyaLimit and Width[1] < BreakLimit and (np.isnan(cat[name])==False or cat[name] > 0):
                Mask.append(True)
            else:
                Mask.append(False)


    if np.sum(Mask) < 3:
        LyaLimit = 1216 * (z + 1)
        BreakLimit = 3000 * (z + 1)
        Mask = []
        for Wave, Width in zip(Cwaves, EffW):
            if Width[0] > LyaLimit and Width[1] < BreakLimit and (np.isnan(cat[name])==False or cat[name] > 0):
                Mask.append(True)
            else:
                Mask.append(False)


    print(filter_names[Mask],cat['f125w_tot_2'])
    return Mask



def FitPowerLaw(photometry,filters,Cwaves,corr,z,Effw,tipo,mu,dmu,ShowPlots=True):

    '''
    photometry: Catalog with photometry
    filters: names of the bands so we can obtain flux and fluxerr from Catalog. Only UV continuum bands, others are masked out previously
    Cwaves : Central wavelengths of filters
    corr : None (not used here)
    z : Redshift
    Effw: Effective width of lines
    tipo : if its photz or specz (makes no difference in this version)
    mu : magnification=1
    dmu : error of magnification = 0
    showplots: If we want to make figures for the resulting power laws of the UV continuum

    '''

    Cwaves=np.array(Cwaves)


    try:
        Flux    = np.array(list(photometry[list(filters[:int(len(filters)/2)])]))
        FluxErr = np.array(list(photometry[filters[int(len(filters)/2):]]))
        for i in range(0,len(FluxErr)):
            if np.isnan(FluxErr[i])==True:
                FluxErr[i]=0.2
    
    except:
        F=[float(photometry[i]) for i in filters]
        Flux=np.array(F[:int(len(filters)/2)])
        FluxErr=np.array(F[int(len(filters)/2):])
        for i in range(0,len(FluxErr)):
            if np.isnan(FluxErr[i])==True:
                FluxErr[i]=0.2



    maskNinetyNines = (Flux>0.0) & (Flux!=-99.0) & (np.isnan(Flux)==False)
    Mags,MagsErr,Cwaves    =   Flux[maskNinetyNines],FluxErr[maskNinetyNines],Cwaves[maskNinetyNines]

    c=2.99792458E+18 #Speed of light
    Flux=(10**(-(Mags+48.6)/2.5))*(c/Cwaves**2) # ABmag to flambda
    flxMinErr=(10**(-(Mags+MagsErr+48.6)/2.5))*(c/Cwaves**2)# ABmag to flambda for lower errorbar
    flxPlusErr=(10**(-(Mags-MagsErr+48.6)/2.5))*(c/Cwaves**2)# ABmag to flambda for upper errorbar
    FluxErr=np.median([flxPlusErr-Flux,Flux-flxMinErr],axis=0)# Average between the two errors, needed because I dont't know how to give asymmetric errors to emcee
    

    
    if len(Mags)<2:
        return np.nan,np.nan,[np.nan,np.nan],[Flux,FluxErr],[[np.nan,np.nan]]
    inValues=[-2,-20] #Initial values of the MCMC for beta and Muv
    labels = ["Beta", "Muv"]
    if tipo=="specz": 
        stp=2000 
        discard=500
        walknum=12
    if tipo=="photz":
        stp=2000
        discard=500
        walknum=12
    print(Flux,FluxErr)
    sampler=MCMCUV.runMCMC([Cwaves,Flux,FluxErr,z,Effw],inValues,tipo,mu,dmu,steps=stp,nwalkers=walknum) #Here is where we run the MCMC

    # Plotting
    if ShowPlots==True and tipo=="specz":
        MCMCUV.plotChain(sampler,labels)
        MCMCUV.plotCorner(sampler,labels,discard=discard)
        MCMCUV.plotModels(sampler,Cwaves,Flux,FluxErr,z,mu,discard=discard,extraPars=z)
    if ShowPlots==True and tipo=="photz":
        MCMCUV.plotChain(sampler,labels)
        MCMCUV.plotCorner(sampler,labels,discard=discard)
        MCMCUV.plotModels(sampler,Cwaves,Flux,FluxErr,z,mu,discard=discard,extraPars=z)
    pairs,betaVals,MuvVals=MCMCUV.returnParameters(sampler,discard=discard)

    betaErr=np.mean([betaVals[1]-betaVals[0],betaVals[2]-betaVals[1]]) #so we just get 1 error bar for our result
    MuvErr=np.mean([MuvVals[1]-MuvVals[0],MuvVals[2]-MuvVals[1]]) 

    return betaVals[1],MuvVals[1],[betaErr,MuvErr],[Flux,FluxErr],pairs


####
#Functions below, we dont really use them for this case
####

def GetDustLaw(Slope):
    if Slope<=-2.23:
        return 1
    elif Slope >-2.23:
        return 10**(1.1*(Slope+2.23)/-2.5) 

def GetLuminosity(z,Muv):
    L = 4*np.pi*(10*u.pc)**2. * 10**(-0.4*(Muv +48.6)) * u.erg/u.s/u.cm**2./u.Hz
    return L.to(u.erg/u.s/u.Hz).value


def GetEW(Flux,Cont,errs):
    val=Flux/Cont
    err=val*(np.sqrt( (errs[0]/Flux)**2 + (errs[1]/Cont)**2  ))
    return val,err


def GetLuminosityHa(z,flux):

    cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
    #flux    =   flux * (2.998e18/(continuumWave**2))
    dL      =   cosmo.luminosity_distance(z).cgs
    Lum     =   flux*(u.erg*u.cm**(-2)*u.s**(-1))*4.0*np.pi*dL**2
    return Lum.value




    ######
    ######

def searchLineLocationHa(lines,z,Cwaves,EffW,FiltNames,FiltNameErr,EffW2,p):
    Waves=dict((n,w) for n,w in zip(FiltNames,Cwaves))
    EFFW=dict((n,ef) for n,ef in zip(FiltNames,EffW2))
    line=lines["Ha"]
    HaFilt,HaFiltErr=[],[]
    #Find filter
    for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
        if c-eff/2 < line*(1+z) <c+eff/2:
            HaFilt.append(name)
            HaFiltErr.append(en)

    line=lines["OiiiHb"]
    OiiiHbFilt,OiiiHbFiltErr=[],[]
    for l in line:
        for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
            if c-eff/2 < l*(1+z) <c+eff/2:
                OiiiHbFilt.append(name)
                OiiiHbFiltErr.append(en)

    
    # Find below 4000A
    FiltersIgnore,IgErr=[],[]
    for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
        if c-eff/2 < 4000*(1+z): 
            FiltersIgnore.append(name)
            IgErr.append(en)

    # Pick continuum filter
    UsedFilts=HaFilt+OiiiHbFilt+FiltersIgnore
    UsedFiltsErr=HaFiltErr+OiiiHbFiltErr+IgErr
    cont,contErr=[],[]
    for n in FiltNames:
        if n not in UsedFilts:
            cont.append(n)
    for n in FiltNameErr:
        if n not in UsedFiltsErr:
            contErr.append(n)

   #### FOR EXCEPTIONS ADD VALUES FOR only Ha or only OiiiHB 
    #Add exceptions for when the redshift does not help obtain the results
    #Global conditions 

#####
    if len(cont)>0:
        temp,temp2=[],[]
        for c,cerr in zip(cont,contErr):
            if p.cat[c]>0.0 and p.cat[cerr]>0.0:
                temp.append(c)
                temp2.append(cerr)
        cont=temp
        contErr=temp2

    if len(cont)==0:
        #print(z,"Not useful")
        return [-99.0,-99.0,-99.0]
    #Ha conditions
    if len(HaFilt)==0:
        #print(z,"Not useful")
        return [-99.0,-99.0,-99.0]
    for ig in FiltersIgnore:
        if ig in HaFilt:
            #print(z,"Not useful")
            return [-99.0,-99.0,-99.0]
    if p.cat[HaFilt][0]<0:
        return [-99.0,-99.0,-99.0]

#####

    cont=[cont[0]]  
    contErr=[contErr[0]]
    c=3e5*u.km/u.s
    wave=(6562*(1+z))*u.Angstrom
    Dif=p.cat[HaFilt[0]]*(c/wave**2)*u.microJansky*EFFW[HaFilt[0]]*u.Angstrom -p.cat[cont[0]]*(c/wave**2)*u.microJansky*EFFW[cont[0]]*u.Angstrom 


    Ha_flambda=(Dif).to(u.erg/u.s/u.cm**2).value
    #print(Ha_flambda)


    Err=np.sqrt( (p.cat[HaFiltErr[0]]*u.microJansky*EFFW[HaFilt[0]]*u.Angstrom)**2  + (p.cat[contErr[0]]*u.microJansky*EFFW[cont[0]]*u.Angstrom  )**2   )/2
    Err=(Err*(c/wave**2)).to(u.erg/u.s/u.cm**2).value

    Ha_flambda=Ha_flambda*0.837 #NII and SII
    Err=Err*0.837



    if Dif<0:
        return [-99.0,-99.0,-99.0]
    else:
        pmin=Ha_flambda-Err
        pmax=Ha_flambda+Err
        if pmin<0.0:
            pmin=0.0
        return [Ha_flambda,pmin,pmax]

def searchLineLocationOiiiHb(lines,z,Cwaves,EffW,FiltNames,FiltNameErr,EffW2,p):
    Waves=dict((n,w) for n,w in zip(FiltNames,Cwaves))
    EFFW=dict((n,ef) for n,ef in zip(FiltNames,EffW2))
    line=lines["Ha"]
    HaFilt,HaFiltErr=[],[]
    #Find filter
    for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
        if c-eff/2 < line*(1+z) <c+eff/2:
            HaFilt.append(name)
            HaFiltErr.append(en)

    line=lines["OiiiHb"]
    OiiiHbFilt,OiiiHbFiltErr=[],[]
    for l in line:
        for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
            if c-eff/2 < l*(1+z) <c+eff/2:
                OiiiHbFilt.append(name)
                OiiiHbFiltErr.append(en)

    
    # Find below 4000A
    FiltersIgnore,IgErr=[],[]
    for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
        if c-eff/2 < 4000*(1+z): 
            FiltersIgnore.append(name)
            IgErr.append(en)

    # Pick continuum filter
    UsedFilts=HaFilt+OiiiHbFilt+FiltersIgnore
    UsedFiltsErr=HaFiltErr+OiiiHbFiltErr+IgErr
    cont,contErr=[],[]
    for n in FiltNames:
        if n not in UsedFilts:
            cont.append(n)
    for n in FiltNameErr:
        if n not in UsedFiltsErr:
            contErr.append(n)

   #### FOR EXCEPTIONS ADD VALUES FOR only Ha or only OiiiHB 
    #Add exceptions for when the redshift does not help obtain the results
    #Global conditions 


#####
    if len(cont)>0:
        temp,temp2=[],[]
        for c,cerr in zip(cont,contErr):
            if p.cat[c]>0.0 and p.cat[cerr]>0.0:
                temp.append(c)
                temp2.append(cerr)
        cont=temp
        contErr=temp2


    if len(cont)==0:
        #print(z,"Not useful")
        return [-99.0,-99.0,-99.0]

    #OiiiHb conditions
    if len(OiiiHbFilt)!=2:
        return [-99.0,-99.0,-99.0]

    if OiiiHbFilt[0]!=OiiiHbFilt[1]:
        return [-99.0,-99.0,-99.0]

    for ig in FiltersIgnore:
        if ig in OiiiHbFilt:
            #print(z,"Not useful")
            return [-99.0,-99.0,-99.0]

    OiiiHbFilt=OiiiHbFilt[0]

    if p.cat[OiiiHbFilt]<0:
        return [-99.0,-99.0,-99.0]

#####
    OiiiHbFilt=[OiiiHbFilt]
    cont=[cont[0]]  
    contErr=[contErr[0]]
    c=3e5*u.km/u.s
    wave=(6562*(1+z))*u.Angstrom
    Dif=p.cat[OiiiHbFilt[0]]*(c/wave**2)*u.microJansky*EFFW[OiiiHbFilt[0]]*u.Angstrom -p.cat[cont[0]]*(c/wave**2)*u.microJansky*EFFW[cont[0]]*u.Angstrom 


    OiiiHb_flambda=(Dif).to(u.erg/u.s/u.cm**2).value
    print(OiiiHb_flambda)


    Err=np.sqrt( (p.cat[OiiiHbFiltErr[0]]*u.microJansky*EFFW[OiiiHbFilt[0]]*u.Angstrom)**2  + (p.cat[contErr[0]]*u.microJansky*EFFW[cont[0]]*u.Angstrom  )**2   )/2
    Err=(Err*(c/wave**2)).to(u.erg/u.s/u.cm**2).value


    if Dif<0:
        return [-99.0,-99.0,-99.0]
    else:
        pmin=OiiiHb_flambda-Err
        pmax=OiiiHb_flambda+Err
        if pmin<0.0:
            pmin=0.0
        return [OiiiHb_flambda,pmin,pmax]




def searchCont(lines,z,Cwaves,EffW,FiltNames,FiltNameErr,EffW2,p):
    Waves=dict((n,w) for n,w in zip(FiltNames,Cwaves))
    EFFW=dict((n,ef) for n,ef in zip(FiltNames,EffW2))
    line=lines["Ha"]
    HaFilt,HaFiltErr=[],[]
    #Find filter
    for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
        if c-eff/2 < line*(1+z) <c+eff/2:
            HaFilt.append(name)
            HaFiltErr.append(en)

    line=lines["OiiiHb"]
    OiiiHbFilt,OiiiHbFiltErr=[],[]
    for l in line:
        for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
            if c-eff/2 < l*(1+z) <c+eff/2:
                OiiiHbFilt.append(name)
                OiiiHbFiltErr.append(en)

    
    # Find below 4000A
    FiltersIgnore,IgErr=[],[]
    for c,eff,name,en in zip(Cwaves,EffW,FiltNames,FiltNameErr):
        if c-eff/2 < 4000*(1+z): 
            FiltersIgnore.append(name)
            IgErr.append(en)

    # Pick continuum filter
    UsedFilts=HaFilt+OiiiHbFilt+FiltersIgnore
    UsedFiltsErr=HaFiltErr+OiiiHbFiltErr+IgErr
    cont,contErr=[],[]
    for n in FiltNames:
        if n not in UsedFilts:
            cont.append(n)
    for n in FiltNameErr:
        if n not in UsedFiltsErr:
            contErr.append(n)

   #### FOR EXCEPTIONS ADD VALUES FOR only Ha or only OiiiHB 
    #Add exceptions for when the redshift does not help obtain the results
    #Global conditions 


#####
    if len(cont)>0:
        temp,temp2=[],[]
        for c,cerr in zip(cont,contErr):
            if p.cat[c]>0.0 and p.cat[cerr]>0.0:
                temp.append(c)
                temp2.append(cerr)
        cont=temp
        contErr=temp2

    if len(cont)==0:
        #print(z,"Not useful")
        return [-99.0,-99.0,-99.0]

    cont=[cont[0]]  
    contErr=[contErr[0]]

    c=2.99792458E+18
    wave=Waves[cont[0]]
    #effW=EFFW[cont[0]]

    cont=p.cat[cont[0]]         *10**-6*10**-23*(c/wave**2)
    contErr=p.cat[contErr[0]]   *10**-6*10**-23*(c/wave**2)

    cont=cont
    contErr=contErr

    if cont<0:
        return [-99.0,-99.0,-99.0]
    else:
        pmin=cont-contErr
        pmax=cont+contErr
        if pmin<0.0:
            pmin=0.0
        return [cont,pmin,pmax]

















































































#######
#######
def searchLineLocationHa_test(lines,z,Cwaves,EffW,FiltNames,p):
    Waves=dict((n,w) for n,w in zip(FiltNames,Cwaves))
    line=lines["Ha"]
    HaFilt=[]
    #Find filter
    for c,eff,name in zip(Cwaves,EffW,FiltNames):
        if c-eff/2 < line*(1+z) <c+eff/2:
            HaFilt.append(name)

    line=lines["OiiiHb"]
    OiiiHbFilt=[]
    for l in line:
        for c,eff,name in zip(Cwaves,EffW,FiltNames):
            if c-eff/2 < l*(1+z) <c+eff/2:
                OiiiHbFilt.append(name)
    
    # Find below 4000A
    FiltersIgnore=[]
    for c,eff,name in zip(Cwaves,EffW,FiltNames):
        if c-eff/2 < 4000*(1+z): 
            FiltersIgnore.append(name)

    # Pick continuum filter
    UsedFilts=HaFilt+OiiiHbFilt+FiltersIgnore
    cont=[]
    for n in FiltNames:
        if n not in UsedFilts:
            cont.append(n)

   #### FOR EXCEPTIONS ADD VALUES FOR only Ha or only OiiiHB 
    #Add exceptions for when the redshift does not help obtain the results
    #Global conditions 


#####

    if len(cont)==0:
        print(z,"Not useful")
        return -99.0
    #Ha conditions
    if len(HaFilt)==0:
        print(z,"Not useful")
        return -99.0
    for ig in FiltersIgnore:
        if ig in HaFilt:
            print(z,"Not useful")
            return -99.0
    print(z,OiiiHbFilt,HaFilt,cont)


def searchLineLocationOiiiHb_test(lines,z,Cwaves,EffW,FiltNames,p):
    Waves=dict((n,w) for n,w in zip(FiltNames,Cwaves))
    line=lines["Ha"]
    HaFilt=[]
    #Find filter
    for c,eff,name in zip(Cwaves,EffW,FiltNames):
        if c-eff/2 < line*(1+z) <c+eff/2:
            HaFilt.append(name)

    line=lines["OiiiHb"]
    OiiiHbFilt=[]
    for l in line:
        for c,eff,name in zip(Cwaves,EffW,FiltNames):
            if c-eff/2 < l*(1+z) <c+eff/2:
                OiiiHbFilt.append(name)
    
    # Find below 4000A
    FiltersIgnore=[]
    for c,eff,name in zip(Cwaves,EffW,FiltNames):
        if c-eff/2 < 4000*(1+z): 
            FiltersIgnore.append(name)

    # Pick continuum filter
    UsedFilts=HaFilt+OiiiHbFilt+FiltersIgnore
    cont=[]
    for n in FiltNames:
        if n not in UsedFilts:
            cont.append(n)

   #### FOR EXCEPTIONS ADD VALUES FOR only Ha or only OiiiHB 
    #Add exceptions for when the redshift does not help obtain the results
    #Global conditions 


#####

    if len(cont)==0:
        print(z,"Not useful")
        return -99.0

    #OiiiHb conditions
    if len(OiiiHbFilt)!=2:
        print(z,"Not useful")
        return -99.0

    if OiiiHbFilt[0]!=OiiiHbFilt[1]:
        print(z,"Not useful")
        return -99.0

    for ig in FiltersIgnore:
        if ig in OiiiHbFilt:
            print(z,"Not useful")
            return -99.0

    print(z,cont)

