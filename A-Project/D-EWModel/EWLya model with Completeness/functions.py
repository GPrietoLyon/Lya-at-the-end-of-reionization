import numpy as np

def Likelihood(ew,dEW,A,Wo):
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

def ParameterModel(physParams,mathParams):
    """
    Take in the physical parameter and the constants of the model.
    Returns A or W
    """
    Muv=physParams
    cMuv,cte= mathParams[0],mathParams[1]
    Parameter   =   (Muv*cMuv)  +   cte 
    return Parameter

def Posterior(ew,dew,physParams,mathParamsA,mathParamsW):

    A=ParameterModel(physParams,mathParamsA)
    Wo=ParameterModel(physParams,mathParamsW)

    #probLike=self.Likelihood(ew,A,Wo)
    probLike=[]
    for i in range(0,len(Detection_type)):
        if Detection_type[i]=="LAE":
            p   =   Likelihood(ew[i],dew[i],A[i],Wo[i])
        #if Detection_type[i]=="nonLAE":
        #    p   =   self.LikNoDet(self.wtab,dew[i],A[i],Wo[i])
        
        if np.isnan(p)==True:
            continue

        probLike.append(p)
        
    return np.array(probLike)

def log_prior(theta,physParams=Muv):
    # I can put priors here or in the equation in the BInf object 
    Auv,Ac,Wuv,Wc= theta
    A =   ParameterModel(physParams,[Auv,Ac])
    W =   ParameterModel(physParams,[Wuv,Wc])
    #print(A,W)
    if (A >= 0.).all() and (A <=1.0).all() and (W > 0.).all():# and (W < 500.).all():
        return 0.0 

    return -np.inf

def log_likelihood(theta,y,yerr,physParams=Muv):
    Auv,Ac,Wuv,Wc = theta
    model = Posterior(y,yerr,physParams,[Auv,Ac],[Wuv,Wc]) # Does thus have to be exp, so it gets outside the logaritm?

    return np.sum(np.log(model))


def log_probability(theta, y, yerr):
    DrawnphysParams=Muv
    
    lp = log_prior(theta,physParams=DrawnphysParams)
    if not np.isfinite(lp):
        return -np.inf
    
    lL=log_likelihood(theta, y, yerr,physParams=DrawnphysParams)
    if np.isnan(lL)==True:
        return -np.inf
    else:
        return lp + lL