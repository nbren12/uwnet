import matplotlib.pyplot as plt
import numpy as np
import wave
import common
import sys

with open(sys.argv[1]) as f:
    S = wave.load(f)

maxstep = S['maxstep']
Input_reg = S['Input_reg']

YMAX = 45 # Upper limit for y-axis
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
