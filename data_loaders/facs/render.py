import sys
sys.path.append('/raid/HKU_TK_GROUP/qindafei/motion-diffusion-model/')
from data_loaders.facs.ICT_model import ICT_model
from data_loaders.facs.facs2obj import mediapipe2ict
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


def save(dataset, all_motions, path):
    ict_model = init_ict_model(decimate=2)
    facs = ict_model.faces

    facs = [d[:52].squeeze().transpose(1, 0) for d in all_motions]
    trans = [None for d in all_motions]
    for _idx, (f, t) in enumerate(zip(facs, trans)):
        f, t = dataset.de_normalize(f, t)
        vert_seq_bs,  vert_seq= to_mesh(ict_model, f, t)
        writePC2(path.replace('.npy', f"{_idx:02d}_bs.pc2"), vert_seq_bs, float16=False)
        if t is not None:
            writePC2(path.replace('.npy', f"{_idx:02d}.pc2"), vert_seq, float16=False)


if __name__ == '__main__':
    import pickle
    from data_loaders.facs.facs2obj import expression_bases_ICT, expression_bases_mediapipe, mapping, mapping_list
    data = pickle.load(open('../pkl/normalized/facs.pkl', 'rb'))
    facs, mean, std = data.values()
    facs = np.concatenate(facs, axis=0)
    facs = facs * std + mean
    facs_ICT = facs[..., mapping_list]
    print()