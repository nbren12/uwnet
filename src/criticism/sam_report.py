import os
from os.path import relpath, dirname, join, exists, abspath
import sys
import subprocess

model = sys.argv[1]

folder = dirname(model)
model = relpath(model, start=folder)

sam_directory = abspath(join(folder, 'test'))

# Run SAM with the neural network
if not exists(sam_directory):
    cmd = [sys.executable, '-m', 'src.criticism.run_sam_ic_nn', model]
    stdout_path = join(folder, 'out')
    stderr_path = join(folder, 'err')
    ret = subprocess.call(
        cmd,
        cwd=folder,
        stdout=open(stdout_path, 'w'),
        stderr=open(stderr_path, 'w'))

    #

    if ret != 0:
        print("STDOUT")
        print("------")
        print(open(stdout_path).read())

        print("STDERR")
        print("------")
        print(open(stderr_path).read())

# make the report
cmd = [
    'jupyter', 'nbconvert', '--output-dir', folder, '--execute',
    'notebooks/templates/sam-run.ipynb'
]
os.environ['RUN'] = sam_directory
subprocess.call(cmd)
