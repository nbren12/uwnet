figures = [
    "scary_instability.pdf",
    "spectra_input_vertical_levels.pdf",
    "standing_instability.pdf",
]

base_state_figs = [
    "base_state_dependence/vary_lts.svg",
    "base_state_dependence/vary_q.svg"
]

dropbox = "~/Dropbox/My\ Articles/InProgress/linear_response_function_paper/figs/"

rule dropbox:
    input: figures + base_state_figs
    shell: "cp {input} {dropbox}"

rule all:
    input: 

rule base_state_plots:
    output:  base_state_figs
    shell: "jupyter nbconvert --execute ./base_state_dependence/2.2-paper-plots.ipynb"

rule stability_plots:
    output: "{script}.pdf"
    script: "{wildcards.script}.py"

rule clean:
    shell: "rm {figures} {base_state_figs}"
