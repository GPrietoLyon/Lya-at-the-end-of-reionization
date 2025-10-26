import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
import corner
from IPython.display import display, Math

cosmo   =   FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)

def model(x,z,beta,Muv):
    dL      =   cosmo.luminosity_distance(z).value
    c=2.99792458E+18
    DM=5*(np.log10(dL*10**6) - 1)
    K= 2.5*np.log10(1.+z)
    x_1500=1500.*(1+z)
    model=(c/x**2)*((x/x_1500)**(beta+2)) * 10**(-(Muv+48.6+DM-K)/2.5) #flambda
    return model


def runMCMC(Observed,inValues,tipo,mu,dmu,steps=3000,nwalkers=50):
    """"
    Here we run the MCMC
    ------
    Input:

    Observed: the real data we need to put into the MCMC to compare to model
    inValues: Initial values for the model parameters
    Steps: MCMC steps
    nwalkers: Number of walkers
    -----
    Output:
    Sampler:MCMC object with results

    ------
    remember that in args, we need to expand the Observed manually, dont know how else to do it

    """""
    pos = inValues+ 1e-5 * np.random.randn(nwalkers,len(inValues) )
    nwalkers, ndim = np.shape(pos)
    #print("Filters",Observed[0],Observed[3])
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(Observed[0],Observed[1],Observed[2],Observed[3],tipo,Observed[4],mu,dmu)
    )
    sampler.run_mcmc(pos, steps, progress=True)
    return sampler

def log_likelihood(theta,x, y, yerr,z,mu,dmu):
    '''
    x : observed Wavelengths, as an input its the central wavelengths of the filters
    y : Fluxes of filters
    yerr : Flux err
    z : redshift
    mu,dmu : Not used in this version
    '''

    dL      =   cosmo.luminosity_distance(z).value
    beta, Muv= theta
    c=2.99792458E+18
    DM=5*(np.log10(dL*10**6) - 1)
    K= 2.5*np.log10(1.+z) # In this case I dont add the (beta-1)factor
    x_1500=1500.*(1+z) #In your code im confused on why you have this term as 1500/(1+z)
    

    model=(c/x**2)*((x/x_1500)**(beta+2)) * 10**(-(Muv+48.6+DM-K)/2.5) #flambda

    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2) )

def log_likelihood_zphot(theta,x, y, yerr,z,effW,mu,dmu):
    '''
    x : observed Wavelengths, as an input its the central wavelengths of the filters
    y : Fluxes of filters
    yerr : Flux err
    z : redshift
    mu,dmu : Not used in this version
    '''
    dL      =   cosmo.luminosity_distance(z).value
    beta, Muv= theta
    c=2.99792458E+18
    DM=5*(np.log10(dL*10**6) - 1)
    K= 2.5*np.log10(1.+z)
    x_1500=1500.*(1+z)

    model=(c/x**2)*((x/x_1500)**(beta+2)) * 10**(-(Muv+48.6+DM-K)/2.5) #flambda

    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2) )



def log_prior(theta):
    beta, Muv = theta
    if -7 < beta < 2 and -30 < Muv < -14  :
        return 0.0
    return -np.inf

def log_probability(theta,x,y, yerr,z,tipo,effW,mu,dmu):
    '''
    x : observed Wavelengths, as an input its the central wavelengths of the filters
    y : Fluxes of filters
    yerr : Flux err
    z : redshift
    tipo : if its specz or photz (no difference in this version)
    effw,mu,dmu : Not used in this version
    '''

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    if tipo=="specz":
        return lp + log_likelihood(theta,x, y, yerr,z,mu,dmu)
    if tipo=="photz":
        return lp + log_likelihood_zphot(theta,x, y, yerr,z,effW,mu,dmu)



#######
#Plotting functions
#######





def plotChain(sampler,labels):
    """"
    Here we run plot MCMC chains
    ------
    Input:

    sampler : sampler object returned from runMCMC
    labels : names of parameters

    """""
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

def plotCorner(sampler,labels,discard=1000):
    """"
    Here we run plot MCMC corner
    ------
    Input:

    sampler : sampler object returned from runMCMC
    labels : names of parameters
    discard : steps to discard

    """""
    ndim=len(labels)
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    fig = corner.corner(
        flat_samples, labels=labels
    )
    plt.show()

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        if mcmc[1]>-12:
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        else:
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))


def plotModels(sampler,x,y,yerr,z,mu,discard=700,extraPars=None):
    """"
    Here we run plot Models with the real data
    ------
    Input:

    sampler : sampler object returned from runMCMC
    x : xvalues (i.e. wavelength)
    y : yvalues (i.e. Flux)
    yerr : yvalue errors (i.e. Flux Error)
    discard : steps to discard
    extraPars : Whatever extra parameters we need to make the plots

    """""
    y,yerr=y, yerr
    y,yerr=y,yerr
    flat_samples = sampler.get_chain(discard=discard, thin=10, flat=True)
    pair=[[*p] for p in flat_samples]

    x0 = np.linspace(min(x),max(x), 1000)
    #for pa in pair:
    #    plt.plot(np.log10(x0),model(x0,z,*[0.652,-21.7]),color="red",alpha=0.2,zorder=0)
    for pa in pair:
        plt.plot(np.log10(x0),model(x0,extraPars,*pa),color="red",alpha=0.05,zorder=0)


    plt.errorbar(np.log10(x), y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel(r"f$_{/nu}$ [erg s cm2 Hz]")
    plt.xlabel("log(Wavelength [A] )")
    plt.show()

def returnParameters(sampler,discard=700):
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    groups=[[*p] for p in flat_samples]
    Vals=[]
    for i in range(len(flat_samples[0])):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        Vals.append(mcmc)
    return groups,*Vals

#

##################
###################




############
#From old polyfits
#############

def ML_fit(x,y,yerr,z,initParams):
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array(initParams) + 0.1 * np.random.randn(len(initParams))
    soln = minimize(nll, initial, args=(x,y, yerr,z))
    return soln.x

def PowerLaw(x, beta, c):
    """
    Not really a power law since we are at log scale
    """
    return x*beta +c 


def normalFit(x,y,yerr,model,Degrees=1,ShowPlots=True):
    """"
    Always trusty polyfit
    ------
    Input:
    x : xvalues (i.e. wavelength)
    y : yvalues (i.e. Flux)
    yerr : yvalue errors (i.e. Flux Error)
    Degrees : Degree of the fit

    """""
    try:
        popt,pcov=np.polyfit(x,y,deg=Degrees,w=1/yerr,cov=True)
    except:
        popt=np.polyfit(x,y,deg=Degrees,w=1/yerr)
        print("ERROR")
        pcov=[[0.0,0.0],[0.0,0.0]]

    if ShowPlots    ==  True:
        plt.errorbar(x,y,yerr=yerr,fmt=".")
        plt.plot(x,10**model(x,*popt))
        #plt.gca().invert_yaxis()
        plt.title("beta: "+ str(popt[0]-2))
        plt.ylabel("log Flux")
        plt.xlabel("log(Wavelength)")
        plt.show()

    return popt[0],popt[1],[np.sqrt(np.diag(pcov))]
