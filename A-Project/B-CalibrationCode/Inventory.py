 #this class will do sorting things and making sample
from Spectra import *



class Inventory:
    def __init__(self,spectra):
        """
        Builder, takes a spectra object
        """
        self.spectra = spectra

    def addSpectra(self,listSpectra,spec):
        """
        Saves spectra into a new list
        Args:
            listSpectra: New list where the spectra objects will be saved
            spec: Spectra object to be saved in the list
        """
        spec = Spectra(spec,self.spectra.Data) # I should improve this line, not necesary to repeat Spectra keys
        listSpectra.append(spec)
