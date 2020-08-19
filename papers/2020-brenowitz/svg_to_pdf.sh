
svgFiles=$(find . -name '*.svg')

for f in $svgFiles
do
    base=${f%.*}
    inkscape $base.svg --batch-process --export-type=pdf --export-filename=$base.pdf 2> /dev/null
done

