import numpy as np
import matplotlib.pyplot as plt
import warnings
class StackSpectra:
    def __init__(self, wave,flux,err):
        """
        wave: All the wavelength arrays of the spectras to be stacked size (Wavelength x Number of spectras)
        flux: All the flux arrays of the spectras to be stacked (Wavelength x Number of spectras)
        """
        self.wave = wave
        self.flux = flux
        self.err = err
        self.central_wave = None


    def maskCentralWave(self, central_wave, width,extractarea):
        """
        Masks the central wavelength of the spectra.

        Parameters:
        - central_wave (float): The central wavelength to be masked.
        - width (float): The width of the mask.

        Returns:
        None
        """
        self.central_wave = central_wave

        extractarea
        for i in range(len(self.wave)):
            mask = (self.wave[i] > central_wave[i] - width/2) & (self.wave[i] < central_wave[i] + width/2) 
            self.flux[i]=self.flux[i][mask]
            self.wave[i]=self.wave[i][mask]
            self.err[i]=self.err[i][mask]
            mask = self.flux[i]<0
            self.flux[i][mask] = 0
            mask=self.wave[i]<(central_wave[i]-extractarea)
            self.flux[i][mask]=0
            mask=self.wave[i]>(extractarea+central_wave[i])
            self.flux[i][mask]=0
        print(self.flux)
        self.width=len(self.flux[i])

    def maskLowSNR(self, snr_threshold):
        """
        Masks the spectra with low signal-to-noise ratio (SNR).

        Parameters:
        - snr_threshold (float): The SNR threshold.

        Returns:
        None
        """
        mask = self.SNR > snr_threshold
        self.flux = self.flux[mask]
        self.err = self.err[mask]
        self.wave = self.wave[mask]
        self.central_wave = self.central_wave[mask]
        self.SNR = self.SNR[mask]

    def loadSNR(self, snr):
        """
        Loads the signal-to-noise ratio (SNR) of the spectra.

        Parameters:
        - snr (array): The SNR array.

        Returns:
        None
        """
        self.SNR = snr

    def calculateSNR_around_10A(self,wavesize=10):
        """
        Calculates the signal-to-noise ratio (SNR) around 10 angstroms of the central wavelength.

        Parameters:
        -

        Returns:
        snr (float): The signal-to-noise ratio.
        """
        SN=[]
        for i in range(len(self.wave)):
            central_wave_10A = self.central_wave[i] - wavesize
            mask = (self.wave[i] > central_wave_10A) & (self.wave[i] < central_wave_10A + wavesize*2)
            flux_10A = self.flux[i][mask]
            err_10A = self.err[i][mask]
            snr = np.sum(flux_10A) / np.sqrt(np.sum(err_10A**2))
            SN.append(snr)
        self.SNR=np.array(SN)


    def plotSpectra(self):
            for i in range(len(self.wave)):
                plt.plot(self.wave[i],self.flux[i])
                plt.fill_between(self.wave[i], -self.err[i],self.err[i],color="gray",alpha=0.3)
                plt.axvline(x=self.central_wave[i])
                plt.title("SNR"+str(self.SNR[i]))
                plt.show()


    def makeArray(self, size=301,resolution=0.62,line_wave=1215.6):
        """
        Creates arrays for wavelength and flux with the specified size.
        This array is where the stack will be made.
        Internally saved in self.Wavelength_array_base and self.Flux_array_base.

        Parameters:
        - size (int): The size of the arrays. Default is 1001.

        Returns:
        None
        """
        start_value = line_wave - (size // 2) * resolution  # calculate the start value
        end_value = line_wave + (size // 2) * resolution  # calculate the end value

        self.Wavelength_array_base = np.linspace(start_value, end_value, size) 
        self.Flux_array_base = np.zeros(size)
        self.Err_array_base = np.zeros(size)

        pass

    def findPeak(self, specialCases=None):
        """
        Takes self.flux and finds the peak of the line.

        Parameters:
        - specialCases (array): Must be an array with [[id1,wavelength],[id2,wavelength],[id3,wavelength]]  Default is None

        Returns:
        None
        """
        peak_indices = []
        for spectrum in self.flux.T:
            peak_index = np.argmax(spectrum)
            peak_indices.append(peak_index)
        
        if specialCases!=None:
            for case in specialCases:
                peak_indices[case[0]] = case[1]

        return peak_indices
    
    def stack(self):
        """
        Stacks the data in self.flux_array_base by their peak.

        Parameters:
        - 
        Returns:
        None
        """
        peak_indices = self.findPeak()
        for i, peak_index in enumerate(peak_indices):
            self.flux[i]=self.flux[i]/np.max(self.flux[i][peak_index])
            self.err[i]=self.err[i]/np.max(self.flux[i][peak_index])


        # Calculate the middle index
        half_width_base = len(self.Flux_array_base) // 2
        half_width_data = self.width // 2

        width_difference=half_width_base-half_width_data
   
        if width_difference<0:
            warnings.warn("Base array width must be bigger than data array")

        # stack
        for f in self.flux:
            for i in range(len(f)):
                self.Flux_array_base[i+width_difference] += f[i]

        for e in self.err:
            for i in range(len(e)):
                self.Err_array_base[i+width_difference] += e[i]**2

        self.stacked_err = np.sqrt(self.Err_array_base)
        self.stacked_spectra=self.Flux_array_base
        self.stacked_wave=self.Wavelength_array_base

        # Shift the peak to the middle
        #self.Flux_array_base = np.roll(self.Flux_array_base, middle_index - peak_indices[0])
        #return self.Flux_array_base





