
svgFiles=$(find . -name '*.svg')

for f in $svgFiles
do
    base=${f%.*}
    inkscape $base.svg --export-pdf=$base.pdf 2> /dev/null
done

