#!/bin/sh
git clone . sambld
cd sambld
docker build -t nbren12/samuwgh .
cd ..
rm -rf sambld
