#!/usr/bin/env python3
import sklearn
import spherecluster
import kaldi_io
import numpy as np
import ptpython.repl

def read_embeddings(path):
  #very important to have the same
  all_vectors = kaldi_io.read_vec_flt_scp(path)
  utts, embeddings = zip(*all_vectors)
  return utts, embeddings

def normalize_embeddings(embeddings):
  embeddings_normalized = []
  for embedding in embeddings:
    norm = np.sum(embedding**2)
    embeddings_normalized.append(embedding/norm)
  return np.array(embeddings_normalized)

def read_utt2spk(path):
  utts = []
  speakers = []
  with open(path) as fi:
    for line in fi:
      utt, spk = line.strip().split()
      utts.append(utt)
      spk = spk[:spk.index("-")] #HACK to actually get speaker id from Tedlium train 
      speakers.append(spk)
  return utts, speakers

def speakers_to_ids(speakers):
  num_speakers = len(set(speakers))
  speaker_to_id = {}
  next_id = 0
  ids = [] 
  for speaker in speakers:
    if speaker not in speaker_to_id:
      speaker_to_id[speaker] = next_id
      next_id +=1
    ids.append(speaker_to_id[speaker])
  return ids

def get_spherical_clustering(embeddings, num_speakers):
  skm = spherecluster.SphericalKMeans(n_clusters = num_speakers)
  skm.fit(embeddings)
  return skm.labels_

def compute_ari(speaker_labels, clustering):
  return sklearn.metrics.adjusted_rand_score(speaker_labels, clustering)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser("Clusters embeddings with Spherical K-means, then computes that cluster's adjusted rand index compared to true speaker identities.")
  parser.add_argument("embeddings", help = "Kaldi scp for embeddings.")
  parser.add_argument("utt2spk", help = "Kaldi utt2spk for true speaker identities.")
  parser.add_argument("--num-rounds", type = int, help = "Number of spherical K-means rounds to average the ARI", default = 50)
  args = parser.parse_args()
  emb_utts, embeddings = read_embeddings(args.embeddings)
  embeddings = normalize_embeddings(embeddings)
  utts, speakers = read_utt2spk(args.utt2spk)
  assert(len(utts) == len(emb_utts))
  assert(a==b for a, b in zip(emb_utts, utts))
  ids = speakers_to_ids(speakers)
  num_speakers = len(set(speakers))
  assert(num_speakers == len(set(ids)))
  aris = []
  for i in range(args.num_rounds):
    clustering_labels = get_spherical_clustering(embeddings, num_speakers)
    aris.append(compute_ari(ids, clustering_labels))
  print("Mean ARI:", np.mean(aris), "Max ARI:", np.max(aris), "Computed over", args.num_rounds, "rounds")

