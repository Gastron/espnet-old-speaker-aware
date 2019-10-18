#!/bin/bash

. ./path.sh
. ./cmd.sh

stage=7 #Start from actually training the network, the data is done already
tag=1a
just_optimize=true #when true, only extract xvectors for dev set

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

xvec_extractor_dir=exp/xvec_${tag}
mkdir -p $xvec_extractor_dir
train_set=train_si284_xvec_local
recog_set="test_dev93_xvec_local test_eval92_xvec_local"

if [ ${stage} -le 0 ]; then
  utils/copy_data_dir.sh data/train_si284 data/${train_set}
  utils/copy_data_dir.sh data/test_dev93 data/test_dev93_xvec_local
  utils/copy_data_dir.sh data/test_eval92 data/test_eval92_xvec_local
  #Train:
  rm -f data/${train_set}/feats.scp
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/xvec_mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/${train_set} exp/make_mfcc/${train_set} mfcc 
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/${train_set} exp/make_vad/${train_set} mfcc
  utils/fix_data_dir.sh data/${train_set}
fi

if [ ${stage} -le 1 ]; then
  #recog_sets
  for name in $recog_set; do
    rm -f data/${name}/feats.scp
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/xvec_mfcc.conf --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc/${name} mfcc 
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_vad/${name} mfcc
    utils/fix_data_dir.sh data/${name}
  done
  exit 0
fi

if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/${train_set} data/${train_set}_no_sil exp/${train_set}_no_sil
  utils/fix_data_dir.sh data/${train_set}_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/${train_set}_no_sil/utt2num_frames data/${train_set}_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/${train_set}_no_sil/utt2num_frames.bak > data/${train_set}_no_sil/utt2num_frames
  utils/filter_scp.pl data/${train_set}_no_sil/utt2num_frames data/${train_set}_no_sil/utt2spk > data/${train_set}_no_sil/utt2spk.new
  mv data/${train_set}_no_sil/utt2spk.new data/${train_set}_no_sil/utt2spk
  utils/fix_data_dir.sh data/${train_set}_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/${train_set}_no_sil/spk2utt > data/${train_set}_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/${train_set}_no_sil/spk2num | utils/filter_scp.pl - data/${train_set}_no_sil/spk2utt > data/${train_set}_no_sil/spk2utt.new
  mv data/${train_set}_no_sil/spk2utt.new data/${train_set}_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/${train_set}_no_sil/spk2utt > data/${train_set}_no_sil/utt2spk

  utils/filter_scp.pl data/${train_set}_no_sil/utt2spk data/${train_set}_no_sil/utt2num_frames > data/${train_set}_no_sil/utt2num_frames.new
  mv data/${train_set}_no_sil/utt2num_frames.new data/${train_set}_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/${train_set}_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
local/nnet3/xvector/tuning/run_xvector_${tag}.sh --stage $stage --train-stage -1 \
  --data data/${train_set}_no_sil --nnet-dir $xvec_extractor_dir \
  --egs-dir exp/xvec_1a/egs

#NOTE if just_optimize=true, script ends here:
if [ $stage -le 9 ] && [ $just_optimize == true ]; then
  #NOTE: dev set is hardcoded here.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
    $xvec_extractor_dir data/test_dev93_xvec_local \
    $xvec_extractor_dir/xvectors_test_dev93_xvec_local
  exit 0
fi

if [ $stage -le 9 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    $xvec_extractor_dir data/${train_set} \
    $xvec_extractor_dir/xvectors_${train_set}

  # Extract x-vectors used in the evaluation.
  for name in $recog_set; do
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
      $xvec_extractor_dir data/${name} \
      $xvec_extractor_dir/xvectors_${name}
  done
fi


