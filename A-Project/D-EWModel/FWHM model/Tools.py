import numpy as np
from scipy.stats import rv_continuous
from decimal import Decimal

def angstromTokms(centralA,As):
    c=300000
    v=c*(As-centralA)/centralA
    return v+c


def lin_equ(l1, l2):
    """Line encoded as l=(x,y)."""
    m = Decimal((l2[1] - l1[1])) / Decimal(l2[0] - l1[0])
    c = (l2[1] - (m * l2[0]))
    return m, c

def Give_instrumental_Res(wave):
    m,c=lin_equ([5800,40],[10000,20])
    return wave*float(m)+float(c)

def DeltaDirac(x):
    if x == 0:
        return 1
    else:
        return 0
    
def DeltaFunc(ew_list):
    new_list = [DeltaDirac(ew) for ew in ew_list]
    return np.array(new_list)
    
def RandomSampler(w):
    #RAND=np.random.uniform(low=0, high=1,size=1)
    #if RAND[0] < (1-a):
    #    return 0.0

    def OriginalLikelihood(ew,Wo=w):
        return ((1/Wo)*np.exp(-ew/Wo)*np.heaviside(ew,0.0))

    class CustomDistribution(rv_continuous):
        def _pdf(self, x):
            # Replace 'funcion' with your actual PDF function
            return OriginalLikelihood(x)
    bmax=1000
    random_values=bmax
    while random_values>=bmax-10:
        custom_dist = CustomDistribution(a=0, b=bmax, name='custom_dist')
        custom_dist._pdf = np.vectorize(OriginalLikelihood)
        random_values = custom_dist.rvs(size=1)
    
    return random_values[0]
