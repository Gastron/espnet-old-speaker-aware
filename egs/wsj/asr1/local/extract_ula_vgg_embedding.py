#!/usr/bin/env python3
# This script will just extract speaker embeddings with Tuomas Kaseva's 
# speaker identification model, if you don't need diarisation
# author: Aku Rouhe

import numpy as np
import librosa
import keras
import scipy.io.wavfile
import subprocess
import io
import numpy as np
import argparse
import sys
import xie_nagrani.model as model

params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

def load_model(path):
    #It's ugly but this should work, I copied it from the reference implementation.
    params = {'dim': (257, None, 1),
                  'nfft': 512,
                  'spec_len': 250,
                  'win_length': 400,
                  'hop_length': 160,
                  'n_classes': 5994,
                  'sampling_rate': 16000,
                  'normalize': True,
                  }
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
    # set up network configuration.
    parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
    parser.add_argument('--ghost_cluster', default=2, type=int)
    parser.add_argument('--vlad_cluster', default=8, type=int)
    parser.add_argument('--bottleneck_dim', default=512, type=int)
    parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
    args = parser.parse_args("")
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='eval', args=args)
    network.load_weights("xie_nagrani/weights.h5", by_name = True)
    return network

def load_audio(rxfilename):
    if rxfilename.lower().endswith(".wav"):
        rate, signal = scipy.io.wavfile.read(rxfilename)
    elif rxfilename.endswith("|"):
        rxfile = rxfilename.split()[:-1] #format and remove |
        out = subprocess.run(rxfile, 
                stdout = subprocess.PIPE)
        filelike = io.BytesIO(out.stdout)
        rate, signal = scipy.io.wavfile.read(filelike)
    else:
        raise ValueError("Bad rxfilename: "+rxfilename)
    signal = signal / max(signal) #to floating point, normalize
    return rate, signal

def load_segments(path):
    segments = {}
    with open(path) as fi:
        for line in fi:
            uttid, recid, start, stop = line.strip().split()
            segments.setdefault(recid, []).append((uttid, float(start), float(stop)))
    return segments

def load_wavscp(path):
    wavscp = {}
    with open(path) as fi:
        for line in fi:
            utt_or_rec_id, rxfile = line.strip().split(maxsplit=1)
            wavscp[utt_or_rec_id] = rxfile
    return wavscp

def wavscp_to_features(wavscp):
    for uttid, rxfile in wavscp.items():
        rate, signal = load_audio(rxfile)
        feat = feature_extraction(rate, signal)
        yield uttid, feat

def segments_to_features(segments, wavscp):
    for recid, rxfile in wavscp.items():
        if recid not in segments:
            continue
        rate, signal = load_audio(rxfile)
        for uttid, start, stop in segments[recid]:
            signal_segment = signal[int(rate*start):int(rate*stop)]
            feat_seq = feature_extraction(rate, signal_segment)
            yield uttid, feat_seq

def feature_extraction(rate, s):
    #STFT
    linear_spect = librosa.stft(s, n_fft=512, win_length=400, hop_length=160)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(mag, 0, keepdims=True)
    std = np.std(mag, 0, keepdims=True)
    spec = (mag - mu)/(std + 1e-5)
    return spec

def extract_embedding(feat_seq, model):
    # Add batch and "channel" dimensions
    feat_seq = feat_seq[np.newaxis, :, :, np.newaxis]
    embedding = model.predict(feat_seq)
    return embedding.flatten()

def features_to_embeddings(features, model):
    for uttid, feat_seq in features:
        embedding = extract_embedding(feat_seq, model)
        yield uttid, embedding

def _write_embs_to(embeddings, fo, fmt):
    for uttid, embedding in embeddings:
        print(uttid, 
                "[ ", 
                " ".join(fmt.format(val) for val in embedding), 
                " ]", 
                file=fo)
def write_embeddings(embeddings, outpath, fmt="{:.8f}"):
    if outpath == "-":
        fo = sys.stdout
        _write_embs_to(embeddings, fo, fmt)
    with open(outpath, "w") as fo:
        _write_embs_to(embeddings, fo, fmt)

if __name__ == "__main__":
    import time
    import pathlib
    parser = argparse.ArgumentParser("Speaker embedding extraction with Xie et.al. model")
    parser.add_argument("--model", help = "path to model to use", 
            default = "./xie_nagrani/weights.h5")
    parser.add_argument("--segments", 
        help = "if should use kaldi style segments, use this option", type = pathlib.Path)
    parser.add_argument("wavscp", help = "data directory to process", type = pathlib.Path)
    parser.add_argument("outfile", help = "where to put the data")
    args = parser.parse_args()
    then = time.time()
    model = load_model(args.model)
    wavscp = load_wavscp(args.wavscp)
    if args.segments is not None:
        print("Segments file exists, using that", file = sys.stderr)
        segments = load_segments(args.segments)
        features = segments_to_features(segments, wavscp)
    else:
        features = wavscp_to_features(wavscp)
    embeddings = features_to_embeddings(features, model)
    write_embeddings(embeddings, args.outfile)
    now = time.time()
    print("Took:", now-then, "seconds", file=sys.stderr)
