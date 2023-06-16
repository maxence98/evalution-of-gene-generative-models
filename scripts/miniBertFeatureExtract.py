from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import random
import numpy as np
import math
import os
from utils import dna
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("models/dnabert-minilm-small", trust_remote_code=True, revision=True)
model = AutoModelForMaskedLM.from_pretrained("models/dnabert-minilm-small", trust_remote_code=True, revision=True).to(device)
model.eval()

def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    return " ".join(kmer)

def seqList2kmerList(seqs, k):
    kmerList = [seq2kmer(seq, k) for seq in seqs]
    return kmerList

def extractFeatures(kmerList, batch_size=64):
    features = np.empty((0, 384))
    for batch_num in range(math.ceil(len(kmerList)/batch_size)):
        batch = kmerList[batch_num*batch_size: (batch_num+1)*batch_size]
        encoded_input = tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_states = outputs.hidden_states[-1].to('cpu')
            batch_features = last_hidden_states[:,0,:].numpy()
            features = np.concatenate([features, batch_features])
    return features

charmap = dna.get_vocab('ATGC')

charlist = [0,0,0,0]
for char in charmap:
    charlist[charmap[char]] = char
def sampleAndConvertGeneratedOneHot2Seqs(sampleFileName, batch_num=2):
    # convert one-hot form into sequence form, return list of sequences
    samples = np.load(sampleFileName)
    samples = samples.transpose((0, 1, 3, 2))
    max_indice = []
    for batch in samples[0:batch_num]:
        for sample in batch:
            max_indice.append(np.argmax(sample, axis=1))
    max_indice = np.array(max_indice)
    #print(max_indice)

    seqs = np.take(charlist, max_indice)
    samples = []
    for seq in seqs:
        samples.append(seq.tobytes().decode().replace('\x00', ''))
    return samples

def extractAllFeatures(experimentPath):
    #experimentPath = '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgangp2007'
    samplesPath = experimentPath + '/samples'
    featuresPath = experimentPath + '/miniFeatures'
    os.makedirs(featuresPath, exist_ok=True)
    sample_names = [f for f in os.listdir(samplesPath) if os.path.isfile(os.path.join(samplesPath, f))]
    for sample_name in sample_names:
        iteration = int(sample_name.split('_')[1].split('.')[0])
        samplesGen = sampleAndConvertGeneratedOneHot2Seqs(os.path.join(samplesPath, sample_name))
        kmerGen = seqList2kmerList(samplesGen, 6)
        featuresGen = extractFeatures(kmerGen)
        np.save(os.path.join(featuresPath, f'{iteration}.npy'), featuresGen)




# Code for generated samples
experimentPaths = [
    #'/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgangp2007',
    #'/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/snwgp5res2203',
    # '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgp2201',
    # '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgp3001',
    # '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgp5res3001',
    # '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/snwgp2201'
    #'/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgps0401',
    #'/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgpm0401',
    # '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/wgp5res32_0401',
    '/data/gpfs/projects/punim1021/yixiangw3/expressionGAN/runs/snwgp5res0401',
]

for experiment in experimentPaths:
    extractAllFeatures(experiment)

# Code for markov samples
# markov_loc = '../data/prom400/markovBaseline'
# markovs = {}
# for k in range(1, 7):
#     with open(os.path.join(markov_loc, f'{k}.txt')) as f:
#         lines = f.read().split('\n')
#     kmerbase = seqList2kmerList(lines, 6)
#     features_base = extractFeatures(kmerbase)
#     np.save(markov_loc + f'/{k}.npy', features_base)


# code for validation samples and baseline
# random.seed(42)
# data_loc = "../data/prom400"
# #data_loc = "../data/human"
# batch_num = 2 # how many batch used for computing distance
# batch_size = 64
# baseline = [''.join(random.choice("ATGC") for _ in range(300)) for _ in range(batch_num * batch_size)]
# kmerbase = seqList2kmerList(baseline, 6)
# features_random = extractFeatures(kmerbase)
# np.save(data_loc + '/miniFeature_random.npy', features_random)


# val_set_loc = data_loc + '/val.txt'

# with open(val_set_loc) as f:
#     lines = f.read().split('\n')
# samplesVal = lines[:batch_size*batch_num]
# samplesVal1 = lines[batch_size*batch_num:2*batch_size*batch_num]

# kmerVal = seqList2kmerList(samplesVal, 6)
# kmerVal1 = seqList2kmerList(samplesVal1, 6)

# features_val = extractFeatures(kmerVal)
# features_val1 = extractFeatures(kmerVal1)

# np.save(data_loc + '/miniFeature_val.npy', features_val)
# np.save(data_loc + '/miniFeature_val1.npy', features_val1)
#print('done')

