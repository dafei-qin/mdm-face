import pickle as pkl
import numpy as np
import os
import random
import torch
from data_loaders.a2m.dataset import Dataset
from glob import glob
from transformers import Wav2Vec2Processor
import soundfile as sf
class biwi_data(Dataset):
    dataname = "biwi"

    def __init__(self, datapath="dataset/biwi", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._pose = {}
        self._num_frames_in_video = {}
        self._actions = {}
        files = glob(os.path.join(datapath, '*au.npy'))
        files = [f.replace('_au.npy', '') for f in files]
        identities = {os.path.basename(f).split('_')[0] for f in files}
        identities = sorted(list(identities))
        fps2ar = 16000 / 25
        _train = {'pose':[], 'au':[], 'name':[], 'iden':[]}
        _val =   {'pose':[], 'au':[], 'name':[], 'iden':[]}
        _test =  {'pose':[], 'au':[], 'name':[], 'iden':[]}


        def load_sequence(fps2ar, iden, iden_sequences, idx, out):
            pose = np.load(iden_sequences[idx] + '.npy').astype(np.float32)
            au = np.load(iden_sequences[idx] + '_au.npy').astype(np.float32)
            frames_pose = len(pose)
            frames_au = len(au) / fps2ar
            # if np.abs(frames_pose - frames_au) > 5: # Drop sequence in which the two sequences length have 5 frames of difference
            #     return out
            min_frames = np.floor(np.min((frames_pose, frames_au))).astype(int)
            min_ar = (min_frames * fps2ar).astype(int)
            out['pose'].append(pose[:min_frames])
            out['au'].append(au[:min_ar])
            out['name'].append(os.path.basename(iden_sequences[idx]).split('_')[-1])
            out['iden'].append(iden)
            return out


        for iden in identities:
            iden_sequences = [f for f in files if iden in f]
            for idx in range(len(iden_sequences) - 4):
                load_sequence(fps2ar, iden, iden_sequences, idx, _train)
            for idx in range(len(iden_sequences) - 4, len(iden_sequences) - 2):
                load_sequence(fps2ar, iden, iden_sequences, idx, _val)
            for idx in range(len(iden_sequences) - 2, len(iden_sequences)):
                load_sequence(fps2ar, iden, iden_sequences, idx, _test)
        
        self._pose = _train['pose'] + _val['pose'] + _test['pose']
        self._au = _train['au'] + _val['au'] + _test['au']
        self._name = _train['name'] + _val['name'] + _test['name']
        self._iden = _train['iden'] + _val['iden'] + _test['iden']
        self._train = np.arange(len(_train['pose']))
        self._val = np.arange(len(_train['pose']), len(_train['pose']) + len(_val['pose']))
        self._test = np.arange(len(_train['pose']) + len(_val['pose']), len(_train['pose']) + len(_val['pose']) + len(_test['pose']))
        self._num_frames_in_video = [len(p) for p in self._pose]
        total_num_actions = len(identities)
        self.num_actions = total_num_actions
        keep_actions = {name:idx for idx, name in enumerate(identities)}
        self._actions = [keep_actions[name] for name in self._iden]
        self._action_classes = {idx:key for idx, key in enumerate(identities)}
        self._action_to_label = keep_actions
        self._label_to_action = {value:key for key, value in keep_actions.items()}
        print(f'Loaded BIWI dataset, total num of sequences: [train]: {len(self._train)}, [val]: {len(self._val)}, [test]: {len(self._test)}')


        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        

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


        inp, action = self._get_verts_data(data_index, frame_ix)
        au, au_raw = self._get_au_data(data_index, frame_ix)

        output = {'inp': inp, 'action': action, 'au': au, 'au_raw': au_raw}

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output
    
    def _get_verts_data(self, data_index, frame_ix):
        pose = self._pose[data_index][frame_ix].transpose(1, 2, 0)
        label = self._actions[data_index]
        return torch.from_numpy(pose), label

    def _get_au_data(self, data_index, frame_ix):
        au_full = self._au[data_index]
        fps2sr = 640 # 16000 / 25
        frame_sr = np.arange(frame_ix.min() * fps2sr, (frame_ix.max() + 1) * fps2sr)
        repeated_frames =  ((frame_ix == frame_ix.max()).astype(int).sum() - 1) * fps2sr
        frame_sr = np.concatenate((frame_sr, [max(frame_sr)] * repeated_frames)).astype(int)
        au_clip = torch.from_numpy(au_full[frame_sr]) # The raw wav data
        au_processed = self.processor(au_clip, sampling_rate=16000).input_values # Data processed for wav2vec2
        return torch.from_numpy(au_processed[0]), au_clip
        