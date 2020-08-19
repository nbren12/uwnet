figures = [
    "scary_instability.pdf",
    "standing_instability.pdf",
]

spectra_figures = [
    "spectra_input_vertical_levels.pdf",
    "spectra_input_vertical_levels_zoom.pdf",
]


base_state_figs = [
    "base_state_dependence/bins-a.pdf",
    "base_state_dependence/bins-b.pdf",
    "base_state_dependence/bins-c.pdf",
    "base_state_dependence/bins-d.pdf"
]

base_state_pdfs = [
    "base_state_dependence/vary_lts.pdf",
    "base_state_dependence/vary_q.pdf"
]

vary_lat_lrf_figs = [
    "base_state_dependence/LRFSVaryLat_in-q_out-q.pdf",
    "base_state_dependence/LRFSVaryLat_in-q_out-s.pdf",
    "base_state_dependence/LRFSVaryLat_in-s_out-q.pdf",
    "base_state_dependence/LRFSVaryLat_in-s_out-s.pdf"
]

maps = [
    "base_state_dependence/q.png",
    "base_state_dependence/lts.png",
]

binned_data_path = "binned.nc"

dropbox = "~/Dropbox/My\ Articles/InProgress/linear_response_function_paper/figs/"

all_figures = figures + spectra_figures + base_state_figs + vary_lat_lrf_figs + maps + base_state_pdfs

rule dropbox:
    input: all_figures
    shell: "cp {input} {dropbox}"
    
rule maps:
    output: maps
    shell: "cd base_state_dependence && python plot_lts_q.py"

rule base_state_plots:
    input: binned_data_path,
           "base_state_dependence/vary_lts.svg",
           "base_state_dependence/vary_q.svg"
    output:  base_state_figs
    shell: "jupyter nbconvert --execute ./base_state_dependence/2.2-paper-plots.ipynb"

rule lrf_zonal_average_vary_lat_plots:
    output:  vary_lat_lrf_figs
    shell: "jupyter nbconvert --execute ./base_state_dependence/3.0-LRF-zonal-averages.ipynb"


rule binned_data:
    output: binned_data_path
    shell: "python base_state_dependence/heating_dependence_on_lts_moisture.py {output}"

rule spectra_plots:
    output: spectra_figures
    script: "spectra_input_vertical_levels.py"

ruleorder: svg_to_pdf > stability_plots

rule svg_to_pdf:
    output: "base_state_dependence/{file}.pdf"
    input: "base_state_dependence/{file}.svg"
    shell: "inkscape {input} --export-pdf={output}"

rule stability_plots:
    output: "{script}.pdf"
    script: "{wildcards.script}.py"

rule clean:
    shell: "rm {figures} {base_state_figs}"
