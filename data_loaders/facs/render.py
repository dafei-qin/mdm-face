import sys
sys.path.append('/raid/HKU_TK_GROUP/qindafei/motion-diffusion-model/')
from data_loaders.facs.ICT_model import ICT_model
from data_loaders.facs.facs2obj import mediapipe2ict
from data_loaders.facs.utils_rot import dof2rot
import numpy as np
from copy import deepcopy
from data_loaders.facs.utils_pc2 import writePC2
def init_ict_model(decimate=0):
    return ICT_model.load(decimate=decimate)



def to_mesh(ict_model, code, trans):
    # Code: (Frames, 53)
    # Trans: (Frames, 4, 4)
    out_v = ict_model.deform(mediapipe2ict(code)) # (Frames, V, 3)
    out_v_bs = deepcopy(out_v)
    if trans is None:
        return out_v_bs, None
    out_v = np.concatenate([out_v, np.ones((out_v.shape[0], out_v.shape[1], 1))], axis=-1) # (Frames, V, 4)
    # out_v = trans[:, np.newaxis] *  out_v
    out_v = (trans[:, np.newaxis] @ out_v[..., np.newaxis])[..., 0]
    out_v = out_v / out_v[..., -1:] # (Frames, V, 4)
    return out_v_bs, out_v[..., :-1]


def save(dataset, all_motions, path, write_names=[]):
    ict_model = init_ict_model(decimate=2)
    facs = ict_model.faces

    facs = [d[:52].squeeze().transpose(1, 0) for d in all_motions]
    trans = [d[52:55] for d in all_motions]
    rot = [d[55:61] for d in all_motions]
    rot = [dof2rot(d) for d in rot]
    trans4x4 = [np.zeros((d.shape[0], 4, 4)) for d in rot] 
    for idx in range(len(trans)):
        trans4x4[idx][:, :3, :3] = rot[idx]
        trans4x4[idx][:, :3, -1] = trans[idx]
        trans4x4[idx][:, -1, -1] = 1

    for _idx, (f, t) in enumerate(zip(facs, trans4x4)):
        if len(write_names) == 0:
            write_name = f"{_idx:02d}"
        else:
            write_name = write_names[_idx]
        f, t = dataset.de_normalize(f, t)
        vert_seq_bs,  vert_seq= to_mesh(ict_model, f, t)
        writePC2(path.replace('.npy', f"{write_name}_bs.pc2"), vert_seq_bs, float16=False)
        if t is not None:
            writePC2(path.replace('.npy', f"{write_name}.pc2"), vert_seq, float16=False)


if __name__ == '__main__':
    import pickle
    from data_loaders.facs.facs2obj import expression_bases_ICT, expression_bases_mediapipe, mapping, mapping_list
    data = pickle.load(open('../pkl/normalized/facs.pkl', 'rb'))
    facs, mean, std = data.values()
    facs = np.concatenate(facs, axis=0)
    facs = facs * std + mean
    facs_ICT = facs[..., mapping_list]
    print()