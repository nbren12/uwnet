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
p = lrf.base_state["pressure"]
rho = lrf.base_state["density"]

structures_to_plot = [
    ("scary_instability.pdf", (1e-6, 50, 1.0)),
    ("standing_instability.pdf", (0.00001, 0.1, 0.2)),
]

for output_file, (k, cp, gr) in structures_to_plot:
    plot_structure(coupler, p, rho, k, phase_speed=cp, growth_rate=gr)
    plt.savefig(output_file)
