"""Snakemake script for fitting a model

Currently supported models are
- 'linear'
- 'mcr'
"""
from lib.models import main, models

if __name__ == '__main__':
    model_type = snakemake.params.model
    main(models[model_type],
         data=snakemake.input[0],
         output=snakemake.output[0])
