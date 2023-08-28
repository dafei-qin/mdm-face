import pickle as pkl
import numpy as np
import os
import random
import torch
from data_loaders.a2m.dataset import Dataset


class biwi_data(Dataset):
    dataname = "biwi"

    def __init__(self, datapath="dataset/biwi", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        self._pose = {}
        self._num_frames_in_video = {}
        self._actions = {}
        for data_type in ['train', 'val', 'test']:
            pkldatafilepath = os.path.join(datapath, f"{data_type}.pkl")
            data = pkl.load(open(pkldatafilepath, "rb"))

            _pose = []
            for key in sorted(data.keys()):
                _pose += data[key]
            self._pose[data_type] = _pose
            # self._pose = [x for data_list in data.values() for x in data_list]
            self._num_frames_in_video[data_type] = [p.shape[0] for p in self._pose[data_type]]
            # self._joints = [self._pose[0].shape[-1]] * len(self._pose) # [nvertices(vertices)/njoints(smpl), 3, nframes]

            _actions = []
            for idx, iden in enumerate(sorted(data.keys())):
                _actions += [idx] * len(data[iden])
            self._actions[data_type] = _actions
            
        total_num_actions = len(data.keys())
        self.num_actions = total_num_actions

        self._train = np.arange(len(self._pose['train']))
        self._val = np.arange(len(self._pose['val'])) + len(self._pose['train'])
        self._test = np.arange(len(self._pose['test'])) + len(self._pose['train']) + len(self._pose['val'])
        
        # Convert back to a single list
        self._pose = [*self._pose['train'], *self._pose['val'], *self._pose['test']]
        self._num_frames_in_video = [*self._num_frames_in_video['train'], *self._num_frames_in_video['val'], *self._num_frames_in_video['test']]
        self._actions = [*self._actions['train'], *self._actions['val'], *self._actions['test']]
        
                                         
        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = {idx:key for idx, key in enumerate(sorted(data.keys()))}

    def __len__(self):
        return len(self._train)
    
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


        output = {'inp': inp, 'action': action}

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output
    
    def _get_verts_data(self, data_index, frame_ix):
        pose = self._pose[data_index][frame_ix].transpose(1, 2, 0)
        label = self._actions[data_index]
        
        return torch.from_numpy(pose), label
    
    # def _load(self, ind, frame_ix):
    #     return super()._load(ind, frame_ix)

actions = {
    0: "no_action"
}
