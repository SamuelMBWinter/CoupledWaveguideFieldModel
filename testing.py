import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.integrate import solve_ivp
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Physical Constants

c = 299.792         # The speed of light  in [um ps^-1]

# Functions

@np.vectorize
def gaussian(x, sigma=0.5, alpha=1, mu=0):
    """
    Standard Gaussian function. Defualt to std. deviation of 1 and amplitude of 1.
    This function is vectorized.

    Parameters:
        x (array type) : values for which you want to calculate the Gaussian function for.

    kwargs:
        sigma (float) : defualts to 1, defines the width of te gaussian peak.
        
        alpha (float) : defaults to 1, defines the amplitudes of the maximum (mean) value.

        mu (float) : defaults to 5, defines the centre of peak and mean value of the function.

    Returns: 
        y (array) : returns an array of the same dimensions of x.
    """

    y = alpha * np.exp(-np.pi*((x-mu)/sigma)**2)
    return y

def DifferenceFucntion(xi, V, omega, s1, s2, p1, p2, r2, kappa, C):
    """
    Difference functions for vectors U_f and U_s combined into V.

    Parameters:
        xi (float) : The current value along the axis the functions have been 
        differentiated in.

        V (array) :  The values of the functions at xi. Here V contains the 
        [U_f, U_s] vectors.

    kwargs:
        s1 (float) : zd/zw

        s2 (float) : 

        p1 (float) : zd/zw

        p2 (float) : 

        r2 (float) : defaults to -1. defined as [] but will always yield +1 or -1.

        kappa (float) : defaults to 0.2, z_d * Deltabeta, the product of the 
        dispersion length and phase mismatch in beta

    Returns:
        dV (array) : The approximate values for the derivatives at each point in V.

    """

    splitV = np.split(V, 3)
    FUf = splitV[0]
    FUs = splitV[1]
    FUp = splitV[2]
    Uf = ifft(FUf, norm="forward")
    Us = ifft(FUs, norm="forward") 

    dFUf = (0-1j) * r2*omega**2 * FUf + (0+1j) * fft(Us*np.conjugate(Uf)* np.exp((0+1j)*kappa *xi), norm="forward")
    
    dFUs = (0-1j)*(s1 * 2 * omega + s2 * omega**2) * FUs + (0+1j) * fft(0.5*Uf**2* np.exp((0-1j)*kappa *xi), norm="forward") + (0-1j) * C * FUp

    dFUp = (0-1j)*(p1 * omega + p2 * omega**2) * FUp + (0-1j) * C * FUs

    dV = np.concatenate([dFUf, dFUs, dFUp])
    return dV

# Classes

class Result:
    def __init__(self, t_0, z_d, length, seperation, tau, omega, xi, V, phase_factors):
        self.__t_0 = t_0
        self.__z_d = z_d
        self.__len = length
        self.__sep = seperation
        self.__tau = tau
        self.__omg = omega
        self.__xi = xi

        FU_f, FU_s, FU_p = np.split(V, 3)
        
        self.__U_f, self.__U_s, self.__U_p = ifft(FU_f, axis=0, norm="forward"), ifft(FU_s, axis=0, norm="forward"), ifft(FU_p, axis=0, norm="forward")
        self.__FU_f, self.__FU_s, self.__FU_p = fftshift(FU_f, axes=0),fftshift(FU_s, axes=0),fftshift(FU_p, axes=0)

        self.__FU_p_phase_adjusted = phase_factors*self.__FU_p
    
    def PrintProperties(self):
        print(f"t_0: \t{self.__t_0}")
        print(f"z_d: \t{self.__z_d}")
        print(f"length: \t{self.__len}")
        print(f"seperation: \t{self.__sep}")

    def PlotU_f(self):
        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_f)**2)
        plt.title('WG1 Fundamental')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\tau$')
        plt.colorbar()
        plt.show()

    def PlotU_s(self):
        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_s)**2)
        plt.title('WG1 Sencond Harmonic')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\tau$')
        plt.colorbar()
        plt.show()

    def PlotU_p(self):
        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_p)**2)
        plt.title('WG2 Pump Field Profile')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\tau$')
        plt.colorbar()
        plt.show()

    def PlotFU_f(self):
        plt.pcolormesh(self.__xi, self.__omg, np.abs(self.__FU_f)**2)
        plt.title('WG1 Fundamental Spectrum')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\omega$')
        plt.colorbar()
        plt.show()

    def PlotFU_s(self):
        plt.pcolormesh(self.__xi, self.__omg, np.abs(self.__FU_s)**2)
        plt.title('WG1 Sencond Harmonic Spectrum')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\omega$')
        plt.colorbar()
        plt.show()

    def PlotFU_p(self):
        plt.pcolormesh(self.__xi, self.__omg, np.abs(self.__FU_p)**2)
        plt.title('WG2 Pump Spectrum')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\omega$')
        plt.colorbar()
        plt.show()

    def PlotAll(self):
        self.PlotU_f()
        self.PlotU_s()
        self.PlotU_p()
        self.PlotFU_f()
        self.PlotFU_s()
        self.PlotFU_p()

    def WriteSpectrum(self, name):
        matrix_dict = {
                "omega": self.__omg,
                "xi": self.__xi,
                "FU_p": self.__FU_p
                }

        savemat(f"{name}.mat", matrix_dict)

    def SaveAllImages(self):
        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_f)**2)
        plt.title('WG1 Fundamental')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\tau$')
        plt.colorbar()
        plt.savefig(f"FundamentalTauXiSep{self.GetSeperation()}Len{self.__len}T100fs.png")
        plt.show()

        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_s)**2)
        plt.title('WG1 Sencond Harmonic')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\tau$')
        plt.colorbar()
        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_p)**2)
        plt.savefig(f"SecondHarmonicTauXiSep{self.GetSeperation()}Len{self.__len}T100fs.png")
        plt.show()

        plt.pcolormesh(self.__xi, self.__tau, np.abs(self.__U_p)**2)
        plt.title('WG2 Pump Field Profile')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\tau$')
        plt.colorbar()
        plt.savefig(f"SPDCPumpTauXiSep{self.GetSeperation()}Len{self.__len}T100fs.png")
        plt.show()

        plt.pcolormesh(self.__xi, self.__omg, np.abs(self.__FU_f)**2)
        plt.title('WG1 Fundamental Spectrum')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\omega$')
        plt.colorbar()
        plt.savefig(f"FundamentalOmegaXiSep{self.GetSeperation()}Len{self.__len}T100fs.png")
        plt.show()

        plt.pcolormesh(self.__xi, self.__omg, np.abs(self.__FU_s)**2)
        plt.title('WG1 Sencond Harmonic Spectrum')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\omega$')
        plt.colorbar()
        plt.savefig(f"SecondHarmoicOmegaXiSep{self.GetSeperation()}Len{self.__len}T100fs.png")
        plt.show()

        plt.pcolormesh(self.__xi, self.__omg, np.abs(self.__FU_p)**2)
        plt.title('WG2 Pump Spectrum')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\omega$')
        plt.colorbar()
        plt.savefig(f"SPDCPumpOmegaXiSep{self.GetSeperation()}Len{self.__len}T100fs.png")
        plt.show()

    def GetSeperation(self):
        return str(self.__sep)



class WaveguidePairModel:
    """
    A class that encapsulates all of the mechanics of the model to more easily test the changes in the physical parameters.

    Defining Parameters:
    t_0 : float
        Approximate pulse width, this is modelled with a gaussian. Units [ps]
    beta_f : float 
        The propogation constant of the fundamental mode in the first waveguide. [um^-1]
    beta_s : float
        The propogation constant of the higher order mode in the first waveguide at twice the frequency of the fundamental.
    beta_p : float 
        The propogation constant of the higer order mode at the same frequency of beta_s but in the second waveguide
    beta_f1: 
    beta_s1: 
    beta_p1: 
    beta_f2: 
    beta_s2: 
    beta_p2: 
    wg_sep : arr (flaots)
        An arrary of waveguide seperations, at which the array of coupling constants are matched.
    coupling_constants : arr (floats)
        The coupling constants at each of the waeguide seperations
    """
    def __init__(self,
            beta_f, 
            beta_s, 
            beta_p, 
            beta_f1, 
            beta_s1, 
            beta_p1, 
            beta_f2, 
            beta_s2, 
            beta_p2, 
            wg_sep, 
            coupling_constants
            ):
        """
        Instantiation function
        """
        self.__beta_f = beta_f
        self.__beta_s = beta_s
        self.__beta_p = beta_p
        self.__beta_f1 = beta_f1
        self.__beta_s1 = beta_s1
        self.__beta_p1 = beta_p1
        self.__beta_f2 = beta_f2
        self.__beta_s2 = beta_s2
        self.__beta_p2 = beta_p2

        self.__coupling_function = interp1d(wg_sep, coupling_constants)


    def RunSimulation(self, t_0, length, seperation, n=512, t_range=[-50, 50], method='DOP853'):
        """
        length : float
            length of the final waveguide in [um]
        seperation : float
            The distance between the waveguides in [nm]
        """
        z_d = 2*t_0**2 / np.abs(self.__beta_f2)

        z_ws = t_0/(self.__beta_s1 - self.__beta_f1)
        z_wp = t_0/(self.__beta_p1 - self.__beta_f1)
        
        kappa = z_d * (self.__beta_s - 2*self.__beta_f)

        s_1 = z_d / z_ws
        p_1 = z_d / z_wp

        r_2 = - z_d * self.__beta_f2 / (t_0**2 * 2)
        s_2 = - z_d * self.__beta_s2 / (t_0**2 * 2)
        p_2 = - z_d * self.__beta_p2 / (t_0**2 * 2)

        normalised_coupling = z_d * self.__coupling_function(seperation)


        nu = z_d * self.__beta_f1 / t_0

        tau_0 = t_range[0] 
        tau_n = t_range[1]  

        tau = np.linspace(tau_0, tau_n, n)

        xi_range = [0, length/z_d] 

        Uf_0 = gaussian(tau, alpha=10)
        Us_0 = np.zeros(n)
        Up_0 = np.zeros(n)
        
        omega = fftfreq(n, tau[1]-tau[0]) 

        FUf_0 = fft(Uf_0, norm="forward")
        FUs_0 = fft(Us_0, norm="forward")
        FUp_0 = fft(Up_0, norm="forward")

        V_0 = np.concatenate([FUf_0, FUs_0, FUp_0]) # V will contain all of the values at each time step

        arguments = (omega, s_1, s_2, p_1, p_2, r_2, kappa, normalised_coupling)

        solution = solve_ivp(DifferenceFucntion, xi_range, V_0, method=method, dense_output=True, args=arguments)#, rtol=10**-7, atol=10**-12) 
        
        xi_axis = np .linspace(xi_range[0], xi_range[1], n)
        omega_axis = fftshift(omega)

        V = solution.sol(xi_axis)

        phase_factor = np.vectorize(lambda x, y: np.exp((0-2j)*np.pi * nu *x*y))
        Xis, Omegas = np.meshgrid(xi_axis, omega_axis)
        phase_factors = phase_factor(Xis, Omegas)

        res = Result(t_0, z_d, length, seperation, tau, omega_axis, xi_axis, V, phase_factors)
        return res

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    import pandas as pd

    t_0 = 0.1                # Approximate pulse duration : ps
    length = 215

    beta_f = 6.7260             # Propogation constant of the fundamental mode in WG1 : um^-1
    beta_s = 12.7539            # Propogation constant of the higher order mode in WG1  
    beta_p = 12.7579            # Propogation constant of the higher order mode in WG2 
    beta_f1 = 0.007734          # Group velocity of the fundamental mode in WG1 : ps um^-1 
    beta_s1 = 0.008836          # Group velocity of the higher order mode in WG1  
    beta_p1 = 0.008723          # Group velocity of the higher order mode in WG2  
    beta_f2 = 8.4535*10**-7     # Dispersion of the fundamental mode in WG1 ps^2 um^-1
    beta_s2 = -2.1754*10**-7    # Dispersion of the higher order mode in WG1
    beta_p2 = 2.620*10**-6     # Dispersion of the higher order mode in WG2 

    df = pd.read_csv("seperationsweep.csv")

    WG_seperation = np.flip(np.array(df["sep"]).flatten())
    C = np.flip(np.array(df["coupling constant"]).flatten())

    Experiment1 = WaveguidePairModel(
                    beta_f, 
                    beta_s, 
                    beta_p, 
                    beta_f1, 
                    beta_s1, 
                    beta_p1, 
                    beta_f2, 
                    beta_s2, 
                    beta_p2, 
                    WG_seperation, 
                    C
                )

    seperation_results = [Experiment1.RunSimulation(t_0, length, s, n=1024, t_range=[-10, 150], method="DOP853") for s in [WG_seperation[0]]]

    for res in seperation_results:
        res.PlotAll()
