import matplotlib.pyplot as plt
import numpy as np

import common

url = 'https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/11_6_GR.pkl'
S = common.load_pickle_from_url(url)

YMAX = 45 # Upper limit for y-axis
Input_reg = np.array([0.01,0.05,0.1,0.15,0.2,0.25])
# Hard-coded table of results from the 4 prognostic tests:
maxstep = np.array([[134,590,446,1499,2044,103], # Orig IC
                  [651,566,332,363,1686,95], # Jan12 IC
                  [512,678,337,840,2011,97], # Jan18 IC
                  [297,504,866,1304,1999,118]]) # Jan24 IC
themean = np.mean(maxstep,axis=0)/48
thestd = np.std(maxstep,axis=0)/48

fig,ax = plt.subplots(figsize=(3.5, 3.5/1.61), constrained_layout=True)
ax2 = ax.twinx()

# Shading goes in the back
ax.fill_between(100*Input_reg,themean-thestd,themean+thestd,color='silver')
ax.plot(100*Input_reg,themean,color='k')
ax.scatter(100*Input_reg,themean,color='k')
ax.set_ylabel('Time to actual\nprognostic failure (days)')
ax.set_xlabel ('Regularization $\sigma$ (%)'); 
ax.set_xlim((-1,26))
ax.set_ylim((0,YMAX))

ax2.semilogy(100*S['Perturbation_std'],1/S['Growth_rate_daym1'],'bo', markeredgecolor='black')
# ax2.set_ylim((0,YMAX))
ax2.set_ylabel('Diagnostic inverse growth rate\nof most unstable mode (days)',color='b')
ax2.spines['right'].set_color('b')
ax2.tick_params(axis='y', colors='b')

plt.savefig('figs/Figure11.pdf', tight=True)
