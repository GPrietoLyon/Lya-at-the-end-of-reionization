import linmix
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class CorrelationObj:
    def __init__(self, x, y,dx,dy,x_uplim=[],y_uplim=[],dx_uplim=[]):
        self.x = x
        self.y = y
        self.dx=dx
        self.dy=dy
        self.x_uplim=x_uplim
        self.y_uplim=y_uplim
        self.dx_uplim=dx_uplim

    def make_uplimFile(self):
        """
        Take data points X,Y, and xuplims,yuplims.
        adds them together into a new x,y file, where the uplims are attached to normal values
        Then generates a list of the same size where a 1 indicates detection and 0 indicates uplim

        Parameters:
        None

        Returns:
        
        None
        """
        Uplim_indicator = list(np.ones(len(self.x)))
        [Uplim_indicator.append(0) for _ in self.x_uplim]
        extra_err=[0.1 for i in self.x_uplim]
        self.new_x = np.array(list(self.x) + list(self.x_uplim))
        self.new_y = np.array(list(self.y) + list(self.y_uplim))
        self.Uplim_indicator=np.array(Uplim_indicator)
        self.new_dx = np.array(list(self.dx) + list(self.dx_uplim))
        self.new_dy = np.array(list(self.dy) + list(extra_err))

    def Run_linmix(self,silent=False,show_results=True):
        self.make_uplimFile()
        lm = linmix.LinMix(self.new_x, self.new_y, self.new_dx, self.new_dy, delta=self.Uplim_indicator)
        lm.run_mcmc(silent=silent)
        if show_results:
            print("{}, {}".format(lm.chain['beta'].mean(), lm.chain['beta'].std()))
            print("{}, {}".format(lm.chain['alpha'].mean(), lm.chain['alpha'].std()))
            print("{}, {}".format(lm.chain['sigsqr'].mean(), lm.chain['sigsqr'].std()))

        return lm.chain['beta'].mean(), lm.chain['beta'].std(), lm.chain['alpha'].mean(), lm.chain['alpha'].std(), lm.chain['sigsqr'].mean(), lm.chain['sigsqr'].std()




