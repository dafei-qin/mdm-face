from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, verts_collate, facs_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == 'biwi':
        from data_loaders.biwi.data.dataset import biwi_data
        return biwi_data
    elif name == 'facs':
        from data_loaders.facs.dataset import facs_data
        return facs_data
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name == 'biwi':
        return verts_collate
    elif name == 'facs':
        return verts_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', data_dir=''):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        if data_dir == '':
            ataset = DATA(split=split, num_frames=num_frames)
        else:
            dataset = DATA(split=split, num_frames=num_frames, datapath=data_dir)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', drop_last=True, data_dir=''):
    dataset = get_dataset(name, num_frames, split, hml_mode, data_dir=data_dir)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=drop_last, collate_fn=collate
    )

    return loader