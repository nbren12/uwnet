#!/usr/bin/env python
"""
Package uwnet into a tarball for distribution

Usage:

python -m uwnet.archive 9_energy_conservation_change_loss/4.pkl uwnet.tar
"""
import sys
import sh
import os

model_file, output = sys.argv[1:]

git_files = sh.git('ls-files').split()
git_files.append(model_file)
git_files = [os.path.join(os.path.relpath(file, ".."))
             for file in git_files]

sh.tar('-C', '..', '-cvf', output, *git_files)
