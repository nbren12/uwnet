from uwnet.spectra import plot_structure_from_path

k = .00001
fig = plot_structure_from_path(
    "../../nn/NNAll/20.pkl",
    structure=(1e-6, 50, 1.0),
)

fig.savefig("scary_instability.pdf")
