import numpy as np
import astropy

def mergeExpositions(exp_file,exp_err,exp_times,Yp):
    '''
    Merges the different expositions of the same slit into a master file

    args:
    exp_file: List of exposures of the same slit/source
    exp_times: list of exposure times (they are already in correct units, no need for dividng by time)

    return

    full: combined exposures weighted by time, units/s 
    '''

    

    
    
    sizes=[ len(e) for e in exp_file]
    maxExp_size=max(sizes)
    maxYpix=Yp[sizes.index(maxExp_size)]
    #print(len(exp_file))


    ListsofExps=[]
    ListsofErrs=[]

    for exp,err,time,pix in zip(exp_file,exp_err,exp_times,Yp):
        full=np.zeros((maxExp_size  , np.shape(exp_file[0])[1]))
        error=np.zeros((maxExp_size , np.shape(exp_file[0])[1]))

        exp=np.nan_to_num(exp)
        err=np.nan_to_num(err)
        pix=(len(exp)-1)-pix

        beggining_dat=int((np.round(maxYpix) - (len(exp) -1 -pix)))
        full[ beggining_dat : beggining_dat + len(exp),:]+=exp ######HERE IM ADDING NANS
        error[ beggining_dat : beggining_dat + len(exp),:]+=(err)**2

        ListsofExps.append(full)
        ListsofErrs.append(error)


    Exps_Clipped = astropy.stats.sigma_clip(ListsofExps, sigma=3, axis=0, maxiters=5)
    Errs_Clipped = astropy.stats.sigma_clip(ListsofErrs, sigma=3, axis=0, maxiters=5)

    #full=np.zeros((maxExp_size  , np.shape(exp_file[0])[1]))
    #for clp in Exps_Clipped:
    #    full+=clp

    #print(np.shape(Exp_Clipped))
    return np.nanmean(Exps_Clipped, axis=0).filled(fill_value=np.nan),np.sqrt(np.nanmean(Errs_Clipped, axis=0).filled(fill_value=np.nan)),maxYpix 




 