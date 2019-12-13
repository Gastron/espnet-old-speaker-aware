#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

stage=0
nnet_dir=exp/xvector_nnet_1a/
test_sets='tedlium-dev tedlium-test'
train_set='tedlium-train-trim'
lda_dim=200

. parse_options.sh

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc



if [ $stage -le 0 ]; then
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
    data/${train_set} exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/${train_set}
  sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
    data/${train_set} exp/make_vad $vaddir
  # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 16 \
    $nnet_dir data/${train_set} \
    data/${train_set}/xvectors
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in $test_sets; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 4 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    # Extract x-vectors used in the evaluation.
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 4 \
      $nnet_dir data/${name} \
      data/${name}/xvectors

  done
fi

if [ $stage -le 2 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd data/$train_set/log/compute_mean.log \
    ivector-mean scp:data/$train_set/xvectors/xvector.scp \
    data/$train_set/xvectors/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality
  $train_cmd data/$train_set/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:data/$train_set/xvectors/xvector.scp ark:- |" \
    ark:data/$train_set/utt2spk data/$train_set/xvectors/lda-transform.mat || exit 1;
fi


