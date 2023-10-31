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

    def __init__(self, datapath="/raid/HKU_TK_GROUP/qindafei/pkl/normalized", inpainting=False, mead=False, **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        self.var_factor = 2
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._pose = {}
        self._num_frames_in_video = {}
        self._actions = {}
        self.mead = False
        if 'mead' in datapath.lower():
            mead = True
            print('Detected MEAD dataset')
        facs, facs_mean, facs_std, names = pickle.load(open(os.path.join(datapath, 'facs.pkl'), 'rb')).values()
        trans, trans_mean, trans_std = pickle.load(open(os.path.join(datapath, 'trans.pkl'), 'rb')).values()
        dof, dof_mean, dof_std = pickle.load(open(os.path.join(datapath, 'dof.pkl'), 'rb')).values()
        if mead:
            self.extracts = pickle.load(open(os.path.join(datapath, 'misc.pkl'), 'rb'))
            self._idens, self._exps, self._levels, self._cameras, self._takes = self.extracts.values()
            self.mead = True
            labels = np.unique(self._exps)
            self._action_to_label = {i:exp.lower() for i, exp in enumerate(labels)}
            self._label_to_action = {exp.lower():i for i, exp in enumerate(labels)}
            self._actions = [self._label_to_action[label.lower()] for label in self._exps]
            self.num_actions = len(labels)
            self._level_to_var = {'level_1': 0.3, 'level_2': 0.6, 'level_3': 0.9}
        # TODO: deal with the misc info and condition the model on 
        # TODO: But first visualize the BS
        # facs, facs_mean, facs_std = pickle.load(open(os.path.join(datapath, 'facs.pkl'), 'rb')).values()
        # trans, trans_mean, trans_std = pickle.load(open(os.path.join(datapath, 'trans.pkl'), 'rb')).values()
        self._pose = [f.astype(np.float32) for f in facs]
        self._names = [os.path.basename(n).replace('.pkl', '') for n in names]
        self._trans = [f.astype(np.float32) for f in trans]
        self._dof = [f.astype(np.float32) for f in dof]
        self._train = np.arange(int(len(self._pose) * 0.9))
        self._val = np.arange(int(len(self._pose) * 0.9), int(len(self._pose) * 0.95))
        self._test = np.arange(int(len(self._pose) * 0.95), len(self._pose))
        self._facs_mean = facs_mean
        self._facs_std = facs_std
        self._trans_mean = trans_mean
        self._trans_std = trans_std
        self._dof_mean = dof_mean
        self._dof_std = dof_std
        if self.mead:
            print(f'Loaded facs (MEAD) dataset, total num of sequences: [train]: {len(self._train)}, [val]: {len(self._val)}, [test]: {len(self._test)}')
        else:
            print(f'Loaded facs (CelebV-HQ) dataset, total num of sequences: [train]: {len(self._train)}, [val]: {len(self._val)}, [test]: {len(self._test)}')

        self._num_frames_in_video = [len(p) for p in self._pose]
        self.inpainting = inpainting


        

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
            
        output = self._get_item_data_index(data_index)
        return output


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
            elif self.sampling == 'disabled':
                frame_ix = np.arange(num_frames)
            else:
                raise ValueError("Sampling not recognized.")


        inp, var, name = self._get_data(data_index, frame_ix)
        action_text = f'{var.item():.2f}'
        output = {'inp': inp, 'var': var, 'action_text': action_text, 'name': name}
        if self.inpainting:
            output['inpainted_motion'] = inp
            output['inpainting_mask'] = torch.zeros_like(inp).bool()
            output['inpainting_mask'][..., :10] = 1
            output['inpainting_mask'][..., -10:] = 1
        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))
        if self.mead:
            output['iden'] = self._idens[data_index]
            output['exp'] = self._exps[data_index]
            output['level'] = self._levels[data_index]
            output['take'] = self._takes[data_index]
            output['camera'] = self._cameras[data_index]
            output['action'] = self._actions[data_index]

        return output
    
    def _get_data(self, data_index, frame_ix):
        pose = self._pose[data_index][frame_ix]
        name = self._names[data_index]
        trans = self._trans[data_index][frame_ix].reshape(-1, 3)
        dof = self._dof[data_index][frame_ix].reshape(-1, 6)
        # trans = self._trans[data_index][frame_ix].reshape(-1, 16)
        pose = np.concatenate((pose, trans, dof), axis=-1)
        if self.mead: # Here we use the level to map to the intensity
            var_level = self._level_to_var[self._levels[data_index]]
            # var_raw = pose.std(axis=0).mean() * self.var_factor
            var = var_level
        else:
            var = pose.std(axis=0).mean()  * self.var_factor
        return torch.from_numpy(pose).transpose(0, 1).unsqueeze(1), torch.tensor([var]).unsqueeze(0), name
    
    def de_normalize(self, facs, trans, dof):
        facs = facs * self._facs_std + self._facs_mean
        if trans is None:
            return facs, None
        
        trans = trans * self._trans_std + self._trans_mean
        if dof is None:
            return facs, trans

        dof = dof * self._dof_std + self._dof_mean
        return facs, trans, dof

    def action_name_to_action(self, label_list):
        return [self._label_to_action[name] for name in label_list]
        