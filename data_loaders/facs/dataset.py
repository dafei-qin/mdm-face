import pickle as pkl
import numpy as np
import os
import random
import torch
from data_loaders.a2m.dataset import Dataset
from glob import glob
from transformers import Wav2Vec2Processor
from tqdm import tqdm
import pickle
class facs_data(Dataset):
    dataname = "facs"

    def __init__(self, datapath="/raid/HKU_TK_GROUP/qindafei/pkl/normalized", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._pose = {}
        self._num_frames_in_video = {}
        self._actions = {}
        facs, facs_mean, facs_std = pickle.load(open(os.path.join(datapath, 'facs.pkl'), 'rb')).values()
        trans, trans_mean, trans_std = pickle.load(open(os.path.join(datapath, 'trans.pkl'), 'rb')).values()
        self._pose = [f.astype(np.float32) for f in facs]
        self._trans = [f.astype(np.float32) for f in trans]
        self._train = np.arange(len(self._pose) - 1000)
        self._val = np.arange(len(self._pose) - 1000, len(self._pose) - 500)
        self._test = np.arange(len(self._pose) - 500, len(self._pose))
        self._facs_mean = facs_mean
        self._facs_std = facs_std
        self._trans_mean = trans_mean
        self._trans_std = trans_std
        print(f'Loaded facs (CelebV-HQ) dataset, total num of sequences: [train]: {len(self._train)}, [val]: {len(self._val)}, [test]: {len(self._test)}')

        self._num_frames_in_video = [len(p) for p in self._pose]


        

    def __len__(self):
        if self.split == 'train':
            return len(self._train)
        else:
            return len(self._val)
    
    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._val[index]
            
        return self._get_item_data_index(data_index)

    def _get_item_data_index(self, data_index):
        
        # Deal with the data length problem
        nframes = self._num_frames_in_video[data_index]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len

            if num_frames > nframes:
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes),
                                               padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")


        inp, var = self._get_data(data_index, frame_ix)

        output = {'inp': inp, 'var': var}

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output
    
    def _get_data(self, data_index, frame_ix):
        pose = self._pose[data_index][frame_ix]
        trans = self._trans[data_index][frame_ix].reshape(-1, 16)
        pose = np.concatenate((pose, trans), axis=-1)
        var = pose.std(axis=0).mean()
        return torch.from_numpy(pose).transpose(0, 1).unsqueeze(1), torch.tensor([var]).unsqueeze(0)
    
    def de_normalize(self, facs, trans):
        facs = facs * self._facs_std + self._facs_mean
        if trans is None:
            return facs, None
        trans = trans * self._trans_std + self._trans_mean
        return facs, trans
