from uwnet.wave.wave import LinearResponseFunction, WaveEq, WaveCoupler
from uwnet.wave.spectra import plot_structure, compute_spectrum
import matplotlib.pyplot as plt


# open lrf
with open("lrf.json") as f:
    lrf = LinearResponseFunction.load(f)

# create a waveeq
wave = WaveEq(lrf.base_state)

# couple them
coupler = WaveCoupler(wave, lrf)

# specific plotting
k = .00001
k, cp, gr = (k, .1, .2)
p = lrf.base_state['pressure']
rho = lrf.base_state['density']

plot_structure(
    coupler, p, rho, k, phase_speed=cp, growth_rate=gr)

plt.savefig("standing_instability.pdf")
