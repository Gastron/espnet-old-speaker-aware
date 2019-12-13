#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

stage=0
extractor_dir=exp/extractor
test_sets='tedlium-dev tedlium-test'
train_set='tedlium-train-trim'
lda_dim=200

. parse_options.sh

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

if [ $stage -le 0 ]; then
  # Make MFCCs and compute the energy-based VAD for the training set
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 32 --cmd "$train_cmd" \
    data/${train_set} exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/${train_set}
  sid/compute_vad_decision.sh --nj 32 --cmd "$train_cmd" \
    data/${train_set} exp/make_vad $vaddir
  utils/fix_data_dir.sh data/${train_set}
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in $test_sets; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
    sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
      $extractor_dir data/$name \
      data/$name/ivectors
  done
fi
exit

if [ $stage -le 4 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 32 \
    $extractor_dir data/$train_set \
    data/$train_set/ivectors
fi

if [ $stage -le 5 ]; then
  for setname in $test_sets; do
    sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
      $extractor_dir data/$setname \
      data/$setname/ivectors
  done
fi

if [ $stage -le 6 ]; then
  echo "Estimating LDA and saving the mean vector"
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd data/$train_set/ivectors/log/compute_mean.log \
    ivector-mean scp:data/$train_set/ivectors/ivector.scp \
    data/$train_set/ivectors/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd data/$train_set/ivectors/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:data/$train_set/ivectors/ivector.scp ark:- |" \
    ark:data/$train_set/utt2spk data/$train_set/ivectors/lda-transform.mat || exit 1;
fi
