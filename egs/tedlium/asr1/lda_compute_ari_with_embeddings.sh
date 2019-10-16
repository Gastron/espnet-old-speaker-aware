#!/bin/bash

utt2spk=data/dev_ivec_local/utt2spk

. ./path.sh
. parse_options.sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <extractor-dir>"
  exit 1
fi

ivec_extractor_dir=$1

ivector-subtract-global-mean $ivec_extractor_dir/ivectors_train_ivec_local/mean.vec \
  scp:$ivec_extractor_dir/ivectors_dev_ivec_local/ivector.scp ark:- |\
  transform-vec $ivec_extractor_dir/ivectors_train_ivec_local/lda-transform.mat \
  ark:- ark,scp:$ivec_extractor_dir/ivectors_dev_ivec_local/ivectors_lda.ark,$ivec_extractor_dir/ivectors_dev_ivec_local/ivectors_lda.scp

source spkemb-venv/bin/activate
python compute_ari_with_embeddings.py $ivec_extractor_dir/ivectors_dev_ivec_local/ivectors_lda.scp $utt2spk

