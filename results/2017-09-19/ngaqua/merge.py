from snakemake.shell import shell

input = snakemake.input
output = snakemake.output

shell("ncks {input[0]} {output}")
for file in input[1:]:
    shell("ncks -A {file} {output}")
