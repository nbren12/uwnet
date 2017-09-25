#!/bin/sh
snakemake  -f cv/ridge/cv.json > test_log
echo "************************************************************"
if $(cmp --silent test_log test_log_true)
then
	echo "test passed"
else
	echo "test failed"
fi
echo "************************************************************"
