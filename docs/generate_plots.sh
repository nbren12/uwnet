#!/bin/sh

files=`ls -d plots.d/*`


for file in $files
do 
    jupyter-nbconvert --execute --inplace $file
done
