import sys
import common
import wave

def pickle_data_to_lrfs():
    dataold = common.load_pickle_from_url("https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/9_13_LRF.pkl")
    dataunstab = common.load_pickle_from_url("https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/2020_03_02_LRF_Unstable.pkl")
    toplot_nam = [
        "MeanLRF_stable",
        "MeanLRF_unstable",
        "MeanLRF_unstable",
        "MeanLRF_unstable",
    ]
    toplot_ind = [
        0,
        1,
        5,
        9,
    ]  # Indices correspond to Perturbation amplitude arrays above
    toplot_tit = ["Stable 1%", "Unstable 1%", "Unstable 10%", "Unstable 20%"]
    lrfs = {}

    for i in range(4):
        if i == 0:
            d = dataold
        else:
            d = dataunstab
        lrf = {
            "base_state": d["base_state"],
            "jacobian": d["linear_response_functions"][toplot_ind[i]][toplot_nam[i]],
        }
        lrfs[toplot_tit[i]] = lrf

    return lrfs

OUTPUT = sys.argv[1]

lrfs = pickle_data_to_lrfs()
with open(OUTPUT, "w") as f:
    wave.dump(lrfs, f)

