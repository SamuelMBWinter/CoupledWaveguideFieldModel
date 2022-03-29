import numpy as np
import matplotlib.pyplot as plt
import Waveguides as wg
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

Experiment1 = wg.WaveguidePairModel(
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

seperation_results = [Experiment1.RunSimulation(t_0, length, s, n=2048, t_range=[-100, 100], method="DOP853") for s in [WG_seperation[0]]]

for res in seperation_results:
    res.PlotAll()
    res.PrintProperties()
#    res.SaveAllImages()
    res.WriteSpectrum(f"FU_pSeperation{res.GetSeperation()}Length{length}t100fs")
