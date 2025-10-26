import numpy as np
import lmfit
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sky_residual(params, sky, data, err):
    """Residual for 1D sky-scaling

    Used to find optimal rescaling of 1D spectrum, A,
    for each spatial pixel

    .. math::

        R_i = \frac{(f_i - A s_i)}{\sigma_i} \\
        \chi^2 = \sum_i R_i^2


    Args:
        params (params): parameters
        sky (ndarray): 1D sky spectrum
        data (ndarray): 3D data cube
        err (ndarray): 1D error spectrum

    Returns:
        residual


    """

    A = params['scale']
    return (data - A * sky)/err

def rescaleSky(skyspec,data,error,header,cont_size=2,skyscale=None):
    """
    Take 1D sky spectrum and rescale each slit by a factor A.
    f_skysub = f - A * Sky
    A is obtained by minimizing f_skysub/std(data)

    Args:
    skyspec: 1D global sky spectrum, only include targets
    data: 2D data
    error: 1D error 

    Returns: f_skysub, corrected by the term of A that better centers the flux distribution at 0.
                                 
    """    

    #####
    #Mask so I only take empty regions
    wave=np.arange(6760.0,6760.0+5631*0.620000004768,0.620000004768)
    MASK = ( (wave > 7638.) & (wave < 7707.) ) | ( (wave > 7893.) & (wave < 7908.) ) | ( (wave > 8106.) & (wave < 8271.) ) \
        | ( (wave > 8314.) & (wave < 8337.) ) | ( (wave > 8469.) & (wave < 8489.) ) | ( (wave > 8508.) & (wave < 8534.) ) \
        | ( (wave > 8552.) & (wave < 8754.) ) | ( (wave > 8795.) & (wave < 8821.) ) | ( (wave > 9006.) & (wave < 9300.) ) \
        | ( (wave > 9571.) & (wave < 9603.) ) | ( (wave > 9625.) & (wave < 9665.) ) | ( (wave > 9749.) & (wave < 9785.) )  
    #####



    if skyscale==None:
        D=[]
        E=[]

        for i in range(0,len(data)):
            if (i<np.round(header['SLITYPIX'])-cont_size or i>np.round(header['SLITYPIX'])+cont_size) and (i>3 and i<len(data)-3):
                D.append(data[i])
                E.append(error[i])      

        sky2d=[]
        for i in range(0,len(D)):
            sky2d.append(skyspec)
        sky2d=np.array(sky2d)

        sky2d=sky2d[:,MASK]

        E=np.array(E)[:,MASK]
        D=np.array(D)[:,MASK]

        params=lmfit.Parameters() 
        params.add('scale', value=1.0)
        out = lmfit.minimize(sky_residual, params, args=(sky2d, D, E),nan_policy="omit")
        skyscale   = out.params['scale'].value 

      


    dat_corr = data - skyscale*skyspec
    dat_corr=dat_corr-np.nanmedian(dat_corr[:,:])


    #print(np.nanmedian(skyspec),np.nanmedian(skyscale*skyspec))
    #print(np.shape(skyspec))
    #plt.hist(data.flatten(),bins=500)
    #plt.show()
    #plt.hist(skyspec,alpha=0.7,bins=500)
    #plt.hist(skyscale*skyspec,alpha=0.7,bins=500)
    #plt.xlim(-0.5e-18,1e-18)
    #plt.show()
    #print(skyspec)
    #plt.plot(skyscale*skyspec*len(data),color="blue",label="A * S")
    #plt.plot(np.nansum(data,axis=0),color="red",label="data")
    #plt.plot(np.nansum(dat_corr,axis=0),label="corrected")
    #print(skyscale)
    #print(data[15][3000],skyspec[3000],dat_corr[15][3000])
    #print(np.nanmean(dat_corr))
    #datcorr_manual=data - skyspec
    #plt.hist(np.array(dat_corr).flatten(),bins=500)
    #plt.axvline(x=np.nanmean(dat_corr),color="blue",label='minimized A')
    #plt.hist(np.array(datcorr_manual).flatten(),color="red",bins=500,alpha=0.6)
    #plt.axvline(x=np.nanmean(datcorr_manual),color="red",label='A=1')
    #plt.xlabel('Flux (All pixels in 2D data)') 
    #plt.legend()
    #plt.xlim(-2e-17,2e-17)
    #plt.show()
    return dat_corr,[skyscale,skyspec]
    