
import common
import wave
import sys
import numpy as np

url = 'https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA/2020_03_02_GR.pkl'
S = common.load_pickle_from_url(url)

S['Input_reg'] = np.array([0.01,0.05,0.1,0.15,0.2,0.25])
# Hard-coded table of results from the 4 prognostic tests:
S['maxstep'] = np.array([[134,590,446,1499,2044,103], # Orig IC
                  [651,566,332,363,1686,95], # Jan12 IC
                  [512,678,337,840,2011,97], # Jan18 IC
                  [297,504,866,1304,1999,118]]) # Jan24 IC


with open(sys.argv[1], "w") as f:
    wave.dump(S, f)