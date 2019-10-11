#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <mdl-dir>"
  echo "Finds and prints all decode results in a model directory"
  echo "NOTE: Does not run scoring!"
  exit 1
fi

for decodedir in $1/decode_*; do
  echo "Directory:"
  echo "$decodedir"
  echo
  grep -e Avg -e SPKR -m 2 ${decodedir}/result.wrd.txt
done
