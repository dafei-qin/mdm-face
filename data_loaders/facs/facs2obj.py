import sys
sys.path.append('/raid/HKU_TK_GROUP/qindafei/motion-diffusion-model/')
from data_loaders.facs.ICT_model import ICT_model
import pickle
from glob import glob
import os
import trimesh
import numpy as np
from data_loaders.facs.utils_pc2 import writePC2
expression_bases_ICT = [
    'browDown_L', 'browDown_R', 'browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'cheekPuff_L', 'cheekPuff_R', 'cheekSquint_L', 'cheekSquint_R', 'eyeBlink_L', 'eyeBlink_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 'eyeLookUp_L', 'eyeLookUp_R', 'eyeSquint_L', 'eyeSquint_R', 'eyeWide_L', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimple_L', 'mouthDimple_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthFunnel', 'mouthLeft', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthPress_L', 'mouthPress_R', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmile_L', 'mouthSmile_R', 'mouthStretch_L', 'mouthStretch_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'noseSneer_L', 'noseSneer_R'
] 

expression_bases_mediapipe = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

double_bases = ['cheekPuff', 'browInnerUp']

mapping = {i:i.replace('_L', 'Left').replace('_R', 'Right') for i in expression_bases_ICT}
mapping['cheekPuff_L'] = 'cheekPuff'
mapping['cheekPuff_R'] = 'cheekPuff'
mapping['browInnerUp_L'] = 'browInnerUp'
mapping['browInnerUp_R'] = 'browInnerUp'

mapping_list = [expression_bases_mediapipe.index(mapping[i]) for i in mapping.keys()]
def mediapipe2ict(facs):
    # Convert the mediapipe FACS to ICT FACS codes
    # facs: (n_frames, 52)
    # return: (n_frames, 53)
    if len(facs.shape) == 1:
        facs = facs[np.newaxis]
    facs_ict = facs[:, mapping_list]
    return facs_ict
        
    
def facs2obj(facs, save_dir, ict_model):
    # Load the ICT model
    
    # Load the FACS codes
    if len(facs.shape) == 1:
        facs = facs[np.newaxis]
    facs_ict = mediapipe2ict(facs)
    out_verts = ict_model.deform(facs_ict)
    faces = ict_model.faces
    writePC2(os.path.join(save_dir, 'facs.pc2'), out_verts)

    
    
if __name__ == '__main__':
    import pickle
    from data_loaders.facs.facs2obj import expression_bases_ICT, expression_bases_mediapipe, mapping, mapping_list
    from data_loaders.facs.render import save
    from data_loaders.facs.dataset import facs_data
    dataset = facs_data()
    data = pickle.load(open('../pkl/normalized/facs.pkl', 'rb'))
    facs, mean, std = data.values()

    save(dataset, [f.transpose(1, 0) for f in facs[:20]], './facs_gt/out.npy')

    print()