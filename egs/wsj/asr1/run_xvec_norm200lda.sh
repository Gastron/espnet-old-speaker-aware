#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.2

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
use_wordlm=false     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=2         # 2 for character LMs
lm_units=650       # 650 for character LMs
lm_opt=adam         # adam for character LMs
lm_batchsize=1024    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=150        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# data
#wsj0=/export/corpora5/LDC/LDC93S6B
#wsj1=/export/corpora5/LDC/LDC94S13B
#Our data is in the other wsj format, see Kaldi wsj recipe
corpus=/m/teamwork/t40511_asr/c/wsj
xvec_base=/scratch/work/rouhea1/kaldi-vanilla/egs/voxceleb/v2/data/

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=train_si284_xvec_norm200lda
dev_set=test_dev93_xvec_norm200lda
test_set=test_eval92_xvec_norm200lda
recog_set="test_dev93_xvec_norm200lda test_eval92_xvec_norm200lda"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    echo "stage 0: Data preparation"
    echo "NOTE: You should have the basic data directories from the normal run.sh already"
    utils/copy_data_dir.sh data/train_si284 data/${train_set}
    utils/copy_data_dir.sh data/test_dev93 data/${dev_set}
    utils/copy_data_dir.sh data/test_eval92 data/${test_set}
    cp data/train_si284/cmvn.ark data/${train_set}
fi

if [ ${stage} -le 2 ]; then
    ln -sf ${xvec_base}/wsj-train-si284/xvectors data/${train_set}/xvectors
    ln -sf ${xvec_base}/wsj-test-dev93/xvectors data/${dev_set}/xvectors
    ln -sf ${xvec_base}/wsj-test-eval92/xvectors data/${test_set}/xvectors
fi

if [ ${stage} -le 4 ]; then
    local/dump_with_xvec.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/xvectors/mean.vec data/${train_set}/xvectors/lda-transform.mat data/${train_set}/xvectors exp/dump_feats/${train_set} ${feat_tr_dir}
    local/dump_with_xvec.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/xvectors/mean.vec data/${train_set}/xvectors/lda-transform.mat data/${dev_set}/xvectors exp/dump_feats/${dev_set} ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/dump_with_xvec.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/xvectors/mean.vec data/${train_set}/xvectors/lda-transform.mat data/${rtask}/xvectors exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}

#if [ ${stage} -le 5 ]; then
#    echo "stage 3: LM Preparation"
#    
#    if [ $use_wordlm = true ]; then
#        lmdatadir=data/local/wordlm_train
#        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
#        mkdir -p ${lmdatadir}
#        cat data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
#        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
#                | grep -v "<" | tr [a-z] [A-Z] > ${lmdatadir}/train_others.txt
#        cat data/${dev_set}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
#        cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
#        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
#        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
#    else
#        lmdatadir=data/local/lm_train
#        lmdict=$dict
#        mkdir -p ${lmdatadir}
#        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
#            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
#        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
#            | grep -v "<" | tr [a-z] [A-Z] \
#            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
#        text2token.py -s 1 -n 1 -l ${nlsyms} data/${dev_set}/text \
#            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
#        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
#                | cut -f 2- -d" " > ${lmdatadir}/test.txt
#        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
#    fi
#
#    # use only 1 gpu
#    if [ ${ngpu} -gt 1 ]; then
#        echo "LM training does not support multi-gpu. signle gpu will be used."
#    fi
#    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
#        lm_train.py \
#        --ngpu ${ngpu} \
#        --backend ${backend} \
#        --verbose 1 \
#        --outdir ${lmexpdir} \
#        --train-label ${lmdatadir}/train.txt \
#        --valid-label ${lmdatadir}/valid.txt \
#        --test-label ${lmdatadir}/test.txt \
#        --resume ${lm_resume} \
#        --layer ${lm_layers} \
#        --unit ${lm_units} \
#        --opt ${lm_opt} \
#        --batchsize ${lm_batchsize} \
#        --epoch ${lm_epochs} \
#        --maxlen ${lm_maxlen} \
#        --dict ${lmdict}
#fi


if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_ivec
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 7 ]; then
    echo "stage 7: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --epochs ${epochs}
fi

if [ ${stage} -le 8 ]; then
    echo "stage 8: Decoding without LM"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} 

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

if [ ${stage} -le 9 ]; then
    echo "stage 9: Decoding with LM"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
        if [ $use_wordlm = true ]; then 
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        if [ $lm_weight == 0 ]; then
            recog_opts=""
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            $recog_opts \
            --lm-weight ${lm_weight} 

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
