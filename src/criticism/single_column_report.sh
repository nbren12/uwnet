#!/bin/bash

if [[ $# < 2 ]]
then
    echo "Usage:"
    echo "single_column_report.sh <model> <data>"
    exit -1
fi

TEMPLATE=notebooks/templates/single-column-tropics.ipynb

export MODEL=$(realpath $1)
export DATA=$(realpath $2)

output_dir=$(dirname reports/$(realpath --relative-to=$(pwd) $MODEL))

mkdir -p $output_dir
jupyter nbconvert --execute --output-dir=$output_dir $TEMPLATE
