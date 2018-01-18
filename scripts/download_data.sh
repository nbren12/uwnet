#!/bin/sh

id=$1
dest=raw/$id

mkdir -p $dest

rsync -avz --progress nbren12@olympus:/home/disk/eos8/nbren12/Data/id/$id/ $dest/
