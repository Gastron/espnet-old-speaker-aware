#!/bin/bash
# Create Speaker Embeddings 

nj=4
cmd="slurm.pl --mem 1G --gpu 1 --time 1:00:00"
embmodel="AkuNet.256.200.2.2000.1000."
extract_script="local/extract_embedding.py"
extra_opts=""

echo "$0 $@"  # Print the command line for logging

. ./path.sh
. ./spkemb-path.sh
. parse_options.sh || exit 1


if [ $# != 1 ]; then
    echo "Usage: $0 datadir"
    exit 1;
fi

data="$1"
logdir="$data"/log
archivedir="$data"/data
# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $archivedir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."

  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/compute_speaker_embeddings_${name}.JOB.log \
    spkemb-venv/bin/python $extract_script --model $embmodel --segments $logdir/segments.JOB $extra_opts $data/wav.scp - \| \
    copy-vector ark,t:- \
    ark,scp:$archivedir/spk_embedding_$name.JOB.ark,$archivedir/spk_embedding_$name.JOB.scp \
    || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/compute_speaker_embeddings_${name}.JOB.log \
    spkemb-venv/bin/python local/extract_embedding.py --model $embmodel $logdir/wav_${name}.JOB.scp - \| \
    copy-vector ark,t:- ark,scp:$archivedir/spk_embedding_$name.JOB.scp \
    || exit 1;

fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $archivedir/spk_embedding_$name.$n.scp || exit 1;
done > $data/embs.scp || exit 1
