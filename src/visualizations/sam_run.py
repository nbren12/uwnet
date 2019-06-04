#!/usr/bin/env python
import os
import sys
import click
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from collections import defaultdict

from src.data import SAMRun, open_ngaqua

IMAGES = defaultdict(list)

report_html = Template(
"""
{% for section, images in sections.items() %}
<h1>{{section}}</h1>
{% for image in images %}
<img src="{{image}}" />
{% endfor %}
{% endfor %}

"""
)

VARS_TO_MAP = ['PW', 'W500', 'NPNN', 'NHNN']
VARS_TO_TROPICS = ['PW', 'W500', 'NPNN', 'NHNN', 'U850', 'V850']

def plot_tropics_mean_pw(case, key):
    z = case.data_2d[key].sel(
        y=slice(4.5e6, 5.5e6), time=slice(None, 110)).mean(['x', 'y'])
    z.plot()


def plot_tropics_avg_cases(runs, key='PW', output=None):
    plt.figure()
    labels = []
    for case, label in runs:
        plot_tropics_mean_pw(case, key)
        labels.append(label)
    plt.legend(labels)
    plt.title(f"Tropical average of {key}")
    if output:
        IMAGES['Tropics Averages'].append(output)
        plt.savefig(output)
        plt.close()


def validate_data(run):
    data_2d = run.data_2d
    start_time = float(data_2d.time.min())
    end_time = float(data_2d.time.max())
    duration  = end_time - start_time

    if duration < 1.0:
        raise ValueError(f"Duration of run is only {duration} days")


def plot_2d_map(run, key, output=None):
    run, name = run
    data_2d = run.data_2d

    start_time = float(data_2d.time.min())
    end_time = float(data_2d.time.max())
    times_to_plot = np.arange(start_time, end_time, 1)

    data = run.data_2d[key].sel(time=times_to_plot)
    data.plot(col='time', col_wrap=3, aspect=2, size=2)
    plt.suptitle(f"{name} {key}")
    if output:
        plt.savefig(output)
        IMAGES['Maps'].append(output)
        plt.close()


def plot_2d_maps(runs, output_dir):
    for key in VARS_TO_MAP:
        plot_2d_map(runs, key, output=f"{output_dir}/{key}.png")


def plot_2d_variables_tropics(run, output_dir):
    for key in VARS_TO_TROPICS:
        plot_tropics_avg_cases(run, key, output=f"{output_dir}/{key}.png")


def relative_paths(paths, output_dir):
    return [os.path.relpath(path, output_dir) for path in paths]


def get_images_relative(output_dir):
    return {section: relative_paths(images, output_dir) for section, images in
            IMAGES.items()}


@click.command()
@click.argument('run_path', type=click.Path())
@click.argument('output_dir', type=click.Path())
@click.option('-c', '--case', type=str, default='control')
def main(run_path, output_dir, case):
    os.mkdir(output_dir)
    ngaqua = open_ngaqua()
    run = SAMRun(run_path, case=case)

    try:
        validate_data(run)
    except ValueError as e:
        print("Exception caught. Exiting gracefully.")
        print(e)
        sys.exit(0)

    if run_path[-1] == '/':
        run_path = run_path[:-1]
    name = os.path.basename(run_path)

    runs = [(run, name), (ngaqua, 'NG-Aqua')]
    plot_2d_variables_tropics(runs, output_dir)

    maps_dir = os.path.join(output_dir, "maps")
    try:
        os.mkdir(maps_dir)
    except FileExistsError:
        pass

    plot_2d_maps((run, name), maps_dir)

    sections = get_images_relative(output_dir)
    with open(f"{output_dir}/index.html", "w") as f:
        html = report_html.render(sections=sections)
        f.write(html)


if __name__ == '__main__':
    main()
