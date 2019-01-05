#!/bin/sh
# Echo the config stored by sacred for a given run id
ID=$1
DB=uwnet

mongo --quiet $DB --eval "\
    db.runs.findOne({_id: $ID}).config
"
