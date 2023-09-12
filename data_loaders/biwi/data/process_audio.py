# Codes adapted from FaceFormer
import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    
from glob import glob
from collections import defaultdict
import argparse

def process_audio(args):

    audio_path = os.path.join(args.audio_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    files = glob(os.path.join(audio_path, '*e*.wav'))
    speech_dict = defaultdict(list)

    for f in sorted(files):
        print(os.path.basename(f))
        
        speech_array, sampling_rate = librosa.load(f, sr=16000)
        speech_dict[os.path.basename(f).split('_')[0]].append(speech_array)
        
    pickle.dump(speech_dict, open(os.path.join(args.save_path, 'audio.pkl', 'wb')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', help='Audio data location')
    parser.add_argument("--save_path", help='save location')
    args = parser.parse_args()
    process_audio(args)

