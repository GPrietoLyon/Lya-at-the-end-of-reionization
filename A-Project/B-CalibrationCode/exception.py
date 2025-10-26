import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-ticks'])

class Exception:
    def __init__(self,ID,LW,RW,CW=None,Skip=None): #Constructor
        """
        Class Builder, this will give us exceptions for the line fitting/MCMC
        
        Args:
            LW: left wavelength limit
            RW: right wavelength limit
            *CW: Central wavelength
        """
        self.ID = ID
        self.LW = LW
        self.RW = RW
        self.CW = CW
        self.Skip = Skip

    def giveW(self):
        dic={}
        dic['ID'] = self.ID
        dic['LW'] = self.LW
        dic['RW'] = self.RW
        dic['CW'] = self.CW
        dic['Skip'] = self.Skip
        return dic

def increaseErr(Yerr):
    constant=3
    newYerr = [ constant*e if e>=np.median(Yerr)+1.5*np.std(Yerr) else 0 for e in Yerr ]
    return newYerr




def giveExceptions():
    """
    For cases that dont behave well, manually set exceptions on the range wavelengths as seen above
    """
    Exc=[]
    #mask1
    Exc.append(Exception('Stark11_43_3982',8148,8165).giveW()) #G
    Exc.append(Exception('z5_GND_44499',8115,8130,8120,True).giveW()) # Skip  
    Exc.append(Exception('z5_GND_7766',8005,8020,8010).giveW())#G
    Exc.append(Exception('z7_GND_8358',9480,9516,9496).giveW()) #G
    Exc.append(Exception('z6_GND_41463',0,0,0,True).giveW()) #skip
    Exc.append(Exception('z6_GNW_18532',0,0,0,True).giveW()) #skip
    Exc.append(Exception('z6_GNW_32543',0,0,0,True).giveW()) #skip
    Exc.append(Exception('Stark11_42_6706',8467,8484,8471).giveW())#G
    Exc.append(Exception('z5_GNW_1503',7343,7365,7353).giveW()) #wont fit Gs //----------------Important-------------//
    Exc.append(Exception('z8_GNW_5376',0,0,0,True).giveW())#skip
    Exc.append(Exception('z6_GNW_8472',0,0,0,True).giveW()) #skip
    Exc.append(Exception('z5_GND_39445',7898,7918,7907).giveW()) #G
    Exc.append(Exception('z6_GND_36100',8015,8050,8028).giveW()) # Gs //----------------Important-------------//
    Exc.append(Exception('z6_GND_35647',7711,7725,7718).giveW()) #Gsh
    Exc.append(Exception('z5_GND_32413',7565,7583,7575).giveW()) #G
    Exc.append(Exception('z6_GND_30340',8258,8290,8272).giveW()) #G
    Exc.append(Exception('z6_GND_19165',8553,8585,8566).giveW()) #G
    Exc.append(Exception('z5_GNW_30237',0,0,0,True).giveW())#skip
    Exc.append(Exception('z5_GNW_29609',7805,7870,7835).giveW()) #Gs//----------------Important-------------//
    Exc.append(Exception('z5_GNW_24858',0,0,0,True).giveW())#skip
    Exc.append(Exception('z5_GNW_25250',0,0,0,True).giveW()) #skip
    Exc.append(Exception('z6_GND_28438',0,0,0,True).giveW()) #WHY IS THIS SOURE TWICE IN THE CATALOG???
    
    #mask1 redone
    Exc.append(Exception('z6_GND_42333',9350,9375,9360,True).giveW()) #Skip
    Exc.append(Exception('z6_GND_7095',9195,9210,9205,True).giveW()) #Skip
    Exc.append(Exception('z7_GND_5323',0,0,0,True).giveW()) #skip  
    Exc.append(Exception('z8_GNW_19912',0,0,0,True).giveW()) #skip  
    Exc.append(Exception('Hu10_z6_4',8125,8140,8133,True).giveW()) #Skip
    Exc.append(Exception('Stark11_35_22248',8120,8140,8130).giveW()) #wont fit


    Exc.append(Exception('z6_GNW_3671',0,0,0,True).giveW()) #Skipped
    Exc.append(Exception('z6_GND_2164',0,0,0,True).giveW()) # Skipped
    Exc.append(Exception('z6_GND_27453',9463,9474,9468,True).giveW()) #SKip
    Exc.append(Exception('z6_GND_34516',0,0,0,True).giveW()) #Skip
    Exc.append(Exception('z5_GND_14430',0,0,0,True).giveW()) #Skipp
    Exc.append(Exception('z6_GNW_30051',0,0,0,True).giveW()) #Skip, weird delta


    #mask2
    Exc.append(Exception('z6_GNW_14478',7950,7992,7960).giveW()) #Recenter mu / DONE / Cant fit ML 
    Exc.append(Exception('z5_GND_464',8345,8375,8353).giveW()) #recenter / Done
    Exc.append(Exception('z5_GND_10047',8515,8530,8520).giveW()) #recenter / Done

    #mask3

    Exc.append(Exception('z5_GNW_22490',7660,7700,7682).giveW()) #recenter /  Done  / Cant fit MCMC
    Exc.append(Exception('z6_GNW_21823',8270,8310,8290).giveW()) #Recenter mu  / PASS / Cant fit ML 

    #mask4

    #No problems


    return Exc

def Double_peaked():
    Ids=[]
    #Ids.append("Jung18_z6_GND_28438")
    #Ids.append("z6_GND_28438")
    Ids.append("Hu10_z7_1")
    Ids.append("z6_GNW_4311")
    Ids.append("z6_GNW_9770")
    Ids.append("z7_GND_11401")

    return Ids
# sigma^2 = sigma^2 +As^2 use master sky to increase errors Done

# Jung18_z6_GND_28438 double peaked? z6_GND_28438
# make broader range Hu10_z7_1 and maybe a wing

# plot sky as a line

#check double solutions

# z6_GND_30340 recheck sky lines
# z6_GNW_30051 check each exposure

#change starting position if skewness is really high

    return Exc

