#!/usr/bin/env python
from pathlib import Path
import click
import sh
import os

@click.command()
@click.argument("path")
def upload_report(path):
    sh.jupyter("nbconvert", path)
    root, ext = os.path.splitext(path)
    html = root + ".html"
    filename = os.path.basename(html)
    url = os.path.join("http://atmos.washington.edu/~nbren12/reports/uw-machine-learning/",
                       filename)

    sh.scp(html, "olympus:~/public_html/reports/uw-machine-learning/")
    print("Report uploaded to {}".format(url))
            

if __name__ == "__main__":
    upload_report()
