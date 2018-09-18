#!/bin/bash

TEMPLATE=notebooks/templates/single-column-tropics.ipynb

export MODEL=$(realpath $1)
export DATA=$(realpath $2)

OUTPUT=$(dirname $MODEL)/single-column-tropics.ipynb
cp $TEMPLATE $OUTPUT

jupyter nbconvert --execute --output-dir=$(dirname $MODEL) $TEMPLATE
