from uwnet.spectra import plot_structure_from_path

k = .00001
fig = plot_structure_from_path(
    "../../nn/NNLowerDecayLR/20.pkl",
    structure=(k, .1, .2),
)

fig.savefig("standing_instability.pdf")
