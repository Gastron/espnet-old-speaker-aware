#!/bin/bash

. ./path.sh
. ./cmd.sh

stage=2 #Start from actually training the network, the data is done already
just_optimize=true #when true, only extract xvectors for dev set
ivec_dim=400
num_gauss=512
num_iter=5
lda_dim=200

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ivec_ubm_dir=exp/ivec_ngauss$num_gauss
ivec_extractor_dir=$ivec_ubm_dir/extractor_dim$ivec_dim
train_set=train_si284_ivec_local
recog_set="test_dev93_ivec_local test_eval92_ivec_local"

if [ ${stage} -le 0 ]; then
  utils/copy_data_dir.sh data/train_si284 data/${train_set}
  utils/copy_data_dir.sh data/test_dev93 data/test_dev93_ivec_local
  utils/copy_data_dir.sh data/test_eval92 data/test_eval92_ivec_local
  #Train:
  rm -f data/${train_set}/feats.scp
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/${train_set} exp/make_mfcc/${train_set} mfcc 
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/${train_set} exp/make_vad/${train_set} mfcc
  utils/fix_data_dir.sh data/${train_set}
fi

if [ $stage -le 1 ]; then
  #recog_sets
  for name in $recog_set; do
    rm -f data/${name}/feats.scp
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc/${name} mfcc 
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_vad/${name} mfcc
    utils/fix_data_dir.sh data/${name}
  done
fi


if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 24 --num-threads 1 \
    data/${train_set} $num_gauss \
    $ivec_ubm_dir/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 24 --remove-low-count-gaussians false \
    data/${train_set} \
    $ivec_ubm_dir/diag_ubm $ivec_ubm_dir/full_ubm
fi

if [ $stage -le 3 ]; then
  # In this stage, we train the i-vector extractor.

  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 16G" \
    --cleanup false \
    --ivector-dim $ivec_dim --num-iters $num_iter \
    $ivec_ubm_dir/full_ubm/final.ubm data/${train_set} \
    $ivec_extractor_dir 
fi


#NOTE if just_optimize=true, script ends here:
if [ $stage -le 4 ] && [ $just_optimize == true ]; then
   sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
     $ivec_extractor_dir data/test_dev93_ivec_local \
     $ivec_extractor_dir/ivectors_test_dev_93_ivec_local

   python compute_ari_with_embeddings.py $ivec_extractor_dir/ivectors_test_dev_93_ivec_local/ivector.scp
   data/test_dev93_ivec_local/utt2spk
   exit 0 
fi

if [ $stage -le 4 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 24 \
    $ivec_extractor_dir data/${train_set} \
    $ivec_extractor_dir/ivectors_${train_set} 

  for name in $recog_set; do
    sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
      $ivec_extractor_dir data/${name} \
      $ivec_extractor_dir/ivectors_$name 
  done
fi
