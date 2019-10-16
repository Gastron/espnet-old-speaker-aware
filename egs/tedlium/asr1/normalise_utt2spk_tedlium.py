#!/usr/bin/env python3

#utt2spks = ["data/tedlium-train-trim/utt2spk",
#    "data/tedlium-dev/utt2spk", 
#    "data/tedlium-test/utt2spk"]
import sys 

with open(sys.argv[1]) as fi:
  for line in fi:
    utt, spk = line.strip().split()
    spk = spk[:spk.index("-")] #HACK to actually get speaker id from Tedlium train 
    print(utt, spk)
