figures = [
    "scary_instability.pdf",
    "spectra_input_vertical_levels.pdf",
    "spectra_input_vertical_levels_zoom.pdf",
    "standing_instability.pdf",
]

base_state_figs = [
    "base_state_dependence/vary_lts.svg",
    "base_state_dependence/vary_q.svg"
]

vary_lat_lrf_figs = [
    "base_state_dependence/LRFSVaryLat_in-q_out-q.pdf",
    "base_state_dependence/LRFSVaryLat_in-q_out-s.pdf",
    "base_state_dependence/LRFSVaryLat_in-s_out-q.pdf",
    "base_state_dependence/LRFSVaryLat_in-s_out-s.pdf"
]

dropbox = "~/Dropbox/My\ Articles/InProgress/linear_response_function_paper/figs/"

rule dropbox:
    input: figures + base_state_figs + vary_lat_lrf_figs
    shell: "cp {input} {dropbox}"

rule base_state_plots:
    output:  base_state_figs
    shell: "jupyter nbconvert --execute ./base_state_dependence/2.2-paper-plots.ipynb"

rule lrf_zonal_average_vary_lat_plots:
    output:  vary_lat_lrf_figs
    shell: "jupyter nbconvert --execute ./base_state_dependence/3.0-LRF-zonal-averages.ipynb"

rule stability_plots:
    output: "{script}.pdf"
    script: "{wildcards.script}.py"

rule clean:
    shell: "rm {figures} {base_state_figs}"
