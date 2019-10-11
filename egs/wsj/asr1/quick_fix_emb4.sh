#!/bin/bash
# Create Speaker Embeddings 

nj=4
cmd="slurm.pl --mem 1G --gpu 1 --time 1:00:00"
model="AkuNet.256.200.2.2000.1000."

echo "$0 $@"  # Print the command line for logging

. ./path.sh
. ./spkemb-path.sh
. parse_options.sh || exit 1


data="data/train_si284_emb/"
logdir="$data"/log
archivedir="$data"/data
# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $archivedir || exit 1;
mkdir -p $logdir || exit 1;

$cmd JOB=4 $logdir/compute_speaker_embeddings_${name}.JOB.log \
  spkemb-venv/bin/python local/extract_embedding.py --model $model $logdir/wav_${name}.JOB.scp - \| \
  copy-vector ark,t:- \
  ark,scp:$archivedir/spk_embedding_$name.JOB.ark,$archivedir/spk_embedding_$name.JOB.scp \
  || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $archivedir/spk_embedding_$name.$n.scp || exit 1;
done > $data/embs.scp || exit 1
