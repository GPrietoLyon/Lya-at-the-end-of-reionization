import matplotlib.pyplot as plt
import numpy as np
#from Tools import *

class PhotoObject:
    def __init__(self,catalog):
        self.cat = catalog
        
        if np.isnan(self.cat["zsys"])==False:
            self.redshift=self.cat["zsys"]
        elif np.isnan(self.cat["z"])==True:
            self.redshift=self.cat["z"]
        else:
            self.redshift=self.cat["photoz"]

    def giveCat(self):
        return self.cat


