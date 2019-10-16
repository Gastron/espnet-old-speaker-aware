#!/usr/bin/env python3
# This script will just extract speaker embeddings with Tuomas Kaseva's 
# speaker identification model, if you don't need diarisation
# author: Aku Rouhe

import numpy as np
import librosa
import keras
import sklearn
import spherecluster
import sys
import scipy.io.wavfile
import subprocess
import io

# This can be changed based on your needs.
SEGMENT_SHIFT = 0.5 #seconds (hop size)

# These parameters need to match the model:
SEGMENT_LENGTH = 2. #seconds
MFCC_SHIFT = 0.01 #seconds
N_MFFC_COMPONENTS = 20

def load_model(modelpath):
    model = keras.models.load_model(modelpath)
    SE_extractor = keras.models.Model(inputs=model.input,
                        outputs=model.layers[-2].output)
    return SE_extractor

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

def ensure_signal_length(signal, rate):
    # Returns (bool, signal), where bool indicates whether
    # zero padding was necessary
    if len(signal) < int(rate*SEGMENT_LENGTH): 
        signal = np.concatenate([signal, 
            np.zeros((int(rate*SEGMENT_LENGTH) - len(signal)), dtype=signal.dtype)])
        return True, signal
    return False, signal

def wavscp_to_features(wavscp):
    for uttid, rxfile in wavscp.items():
        try:
           rate, signal = load_audio(rxfile)
        except ValueError:
            print("ERROR: File", uttid, "resulted in error.",
                    file=sys.stderr)
            continue
        had_to_pad, signal = ensure_signal_length(signal, rate)
        if had_to_pad:
            print("WARNING: File", uttid, "had to be zero padded, too short for embedding",
                    file=sys.stderr)
        feat = feature_extraction(rate, signal)
        yield uttid, feat

def segments_to_features(segments, wavscp):
    for recid, rxfile in wavscp.items():
        if recid not in segments:
            continue
        rate, signal = load_audio(rxfile)
        for uttid, start, stop in segments[recid]:
            signal_segment = signal[int(rate*start):int(rate*stop)]
            had_to_pad, signal_segment = ensure_signal_length(signal_segment, rate)
            if had_to_pad:
                print("WARNING: Segment", uttid, "had to be zero padded, too short for embedding",
                    file=sys.stderr)
            feat_seq = feature_extraction(rate, signal_segment)
            yield uttid, feat_seq

def feature_extraction(rate, sig):
    S = np.transpose(librosa.util.frame(sig, 
        int(rate*SEGMENT_LENGTH), 
        int(rate*SEGMENT_SHIFT)))
    mfcc_feats = []
    for frame in S:
        # MFCC
        mfcc_feat = librosa.feature.mfcc(frame, 
                n_mfcc = N_MFFC_COMPONENTS, 
                sr = rate, 
                n_fft = 512, 
                hop_length = int(rate*MFCC_SHIFT))
        mfcc_feat = sklearn.preprocessing.scale(mfcc_feat, axis = 1)
        # Derivatives
        mfcc_d = librosa.feature.delta(mfcc_feat, mode = "nearest")
        mfcc_d2 = librosa.feature.delta(mfcc_feat, order = 2, mode = "nearest")
        x = np.concatenate([mfcc_feat, mfcc_d, mfcc_d2], axis = 0)
        # Energy removed
        x = np.delete(x, 0, axis = 0)
        mfcc_feats.append(x.T) #want input in [frame-dim, mfcc-sequence-dim, mfcc-coefficent-dim]
    return np.array(mfcc_feats)

def extract_embedding_seq(feat_seq, model):
    return model.predict(feat_seq)

def get_embedding_center(embedding_seq):
    # Extract embedding center as the spherical K-means cluster center
    # with K=1 
    skm = spherecluster.SphericalKMeans(n_clusters = 1)
    skm.fit(embedding_seq)
    center = skm.cluster_centers_[0]
    return center

def features_to_embeddings(features, model):
    for uttid, feat_seq in features:
        embedding_seq = extract_embedding_seq(feat_seq, model)
        center = get_embedding_center(embedding_seq)
        yield uttid, center

def _write_embs_to(embeddings, fo, fmt):
    for uttid, center in embeddings:
        print(uttid, 
                "[ ", 
                " ".join(fmt.format(val) for val in center), 
                " ]", 
                file=fo)
def write_embeddings(embeddings, outpath, fmt="{:.8f}"):
    if outpath == "-":
        fo = sys.stdout
        _write_embs_to(embeddings, fo, fmt)
    with open(outpath, "w") as fo:
        _write_embs_to(embeddings, fo, fmt)

if __name__ == "__main__":
    import argparse
    import time
    import pathlib
    parser = argparse.ArgumentParser("Speaker embedding extraction with Tuomas Kaseva's model")
    parser.add_argument("--model", help = "path to model to use", 
            default = "./EMB_model.250.256.voxceleb.1.2000.625.1")
    parser.add_argument("--segment-shift", type = float, 
            help = "how often to extract an embedding, fractions of seconds",
            default = 0.5)
    parser.add_argument("--segments", 
        help = "if should use kaldi style segments, use this option", type = pathlib.Path)
    parser.add_argument("wavscp", help = "data directory to process", type = pathlib.Path)
    parser.add_argument("outfile", help = "where to put the data")
    args = parser.parse_args()
    SEGMENT_SHIFT = args.segment_shift
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
