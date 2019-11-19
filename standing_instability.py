from uwnet.spectra import plot_structure_from_path
from uwnet.wave import LinearResponseFunction, WaveEq, WaveCoupler
from uwnet.spectra import compute_spectrum


# open lrf
with open("lrf.json") as f:
    lrf = LinearResponseFunction.load(f)


# create a waveeq
wave = WaveEq(lrf.base_state)

# couple them
coupler = WaveCoupler(wave, lrf, lrf.base_state)
eig = compute_spectrum(coupler)

# k = .00001
# fig = plot_structure_from_path(
#     "../../nn/NNLowerDecayLR/20.pkl",
#     structure=(k, .1, .2),
# )

# fig.savefig("standing_instability.pdf")
