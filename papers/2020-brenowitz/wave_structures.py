from uwnet.wave import LinearResponseFunction, WaveEq, WaveCoupler
from uwnet.wave.spectra import plot_structure
import matplotlib.pyplot as plt

def get_wave_coupler(path):

    # open lrf
    with open(path) as f:
        lrf = LinearResponseFunction.load(f)

    # create a waveeq
    wave = WaveEq(lrf.base_state)

    # couple them
    return WaveCoupler(wave, lrf)

# specific plotting

structures_to_plot = [
    ("figs/scary_instability.pdf", "lrf/nn_NNAll_20.json", (1e-6, 44, 1.0)),
    ("figs/standing_instability.pdf","lrf/nn_lower_decay_lr_20.json", (0.00001, 0.1, 0.2)),
]

for output_file, lrf_path, (k, cp, gr) in structures_to_plot:
    coupler = get_wave_coupler(lrf_path)
    p = coupler.lrf.base_state["pressure"]
    rho = coupler.lrf.base_state["density"]
    plot_structure(coupler, p, rho, k, phase_speed=cp, growth_rate=gr)
    plt.savefig(output_file)
