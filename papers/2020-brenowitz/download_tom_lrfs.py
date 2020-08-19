import sys
import common
import wave

S2_URL = "https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/8_19_LRF.pkl"

def pickle_data_to_lrfs() -> dict:
    """tom saved data in pickles files with a different structure

    This function opens all the LRFs he computed into a single dictionary.
    """
    LRF_KEY = 'linear_response_functions'
    JACOBIAN_KEY = 'jacobian'
    BASE_STATE_KEY = 'base_state'

    def download_tom_data_1():
        dataunstab = common.load_pickle_from_url("https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/2020_03_02_LRF_Unstable.pkl")

        # read in unstable data
        lrfs = {}
        for ind, name in [
            (0, 'Unstable'),
            (1, 'Unstable 1%'),
            (5, 'Unstable 10%'),
            (9, 'Unstable 20%'),
        ]:
            lrfs[name] = {
                'base_state': dataunstab[BASE_STATE_KEY],
                JACOBIAN_KEY: dataunstab[LRF_KEY][ind]['MeanLRF_unstable'],
            }

        return lrfs


    def download_tom_data_2():
        name = "MeanLRF_stable"
        title = "Stable"
        d = common.load_pickle_from_url(S2_URL)

        return {title: 
            {
                "base_state": d[BASE_STATE_KEY],
                JACOBIAN_KEY: d[LRF_KEY][name],
            }
        }

    def download_tom_data_3():
        # read in stable data
        lrfs = {}

        dataold = common.load_pickle_from_url("https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/9_13_LRF.pkl")
        lrfs['Stable 1%'] = {
            'base_state': dataold[BASE_STATE_KEY],
            JACOBIAN_KEY: dataold[LRF_KEY][0]['MeanLRF_stable']
        }

        return lrfs


    output = {}
    for data in [download_tom_data_1(), download_tom_data_2(), download_tom_data_3()]:
        output.update(data)

    return output
    
if __name__ == "__main__":

    OUTPUT = sys.argv[1]

    lrfs = pickle_data_to_lrfs()
    with open(OUTPUT, "w") as f:
        wave.dump(lrfs, f)

