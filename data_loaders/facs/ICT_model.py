import sys
import numpy as np
import pickle
from copy import deepcopy
import trimesh
import pickle
import logging
try:
    import face_model_io
    has_ICT_orig = True

except ModuleNotFoundError: 
    has_ICT_orig = False

class ICT_model():
    # Customized ICT model with faster loading speed.
    # ICT-FaceKit has a 100 dim identity space
    # And a 53 dim FACS space
    # We support three different templates:
    # decimate = 0: Original ICT-FaceKit with only head areas
    # decimate = 1: Cut the mouth and nose
    # decimate = 1: Decimated version with only 3k vertices
    def __init__(self, path, decimate=0,  load=None, code=np.zeros((1, 153))):
        self.code = code 
        if load is not None:

            return self.load(load)
        if not has_ICT_orig:
            raise Exception('No ICT-FaceKit module found, Please ensure the module to init from sketch')
            
        face_model =  face_model_io.load_face_model(path)
        self.vertices = trimesh.load('./third_party/ICT-FaceKit/template_only_face.obj', process=False).vertices
        self.faces = trimesh.load('./third_party/ICT-FaceKit/template_only_face.obj', process=False).faces
        self.identity_names = face_model._identity_names
        self.expression_names = face_model._expression_names
        self.identity_weights = face_model._identity_shape_modes[:, :11248]
        self.expression_weights = face_model._expression_shape_modes[:, :11248]
        self.weights = np.concatenate((self.identity_weights, self.expression_weights), axis=0)
        if decimate == 2:
            un_subdivide_mapping = np.load('./third_party/ICT-FaceKit/11248-11089.npy')
            self._load_map(un_subdivide_mapping)
            un_subdivide_mapping = np.load('./third_party/ICT-FaceKit/10089-3694.npy')
            self._load_map(un_subdivide_mapping)
            self.faces = trimesh.load('./third_party/ICT-FaceKit/template_only_face_3694.obj', process=False).faces
        elif decimate == 1:
            un_subdivide_mapping = np.load('./third_party/ICT-FaceKit/11248-11089.npy')
            self._load_map(un_subdivide_mapping)
            self.faces = trimesh.load('./third_party/ICT-FaceKit/template_only_face_10089.obj', process=False).faces

            

    def _load_map(self, referenced):
        self.vertices = self.vertices[referenced]
        self.identity_weights = self.identity_weights[:, referenced]
        self.expression_weights = self.expression_weights[:, referenced]
        self.weights = self.weights[:, referenced]
        


    def deform(self, code):
        # Change the identity & expression of the meshes in a batch processing way.
        B = code.shape[0]
        if code.shape[1] == 53: # Change the expression
            _code = np.concatenate([self.code[:, :100].repeat(B, 0), code], axis=-1)
        elif code.shape[1] == 100: # Change the identity
            _code = np.concatenate([code, self.code[:, 100:].repeat(B, 0)], axis=-1)
        else: # Change both
            _code = code
        out_v = deepcopy(self.vertices[np.newaxis].repeat(B, 0)) * 10 # (B, V, 3)
        out_v += (_code[..., np.newaxis][..., np.newaxis] * self.weights[np.newaxis]).sum(axis=1) # (B, D, 1, 1) x (1, D, V, 3) --> (B, V, 3)
        out_v = out_v * 0.1
        return out_v
    
    def change_iden(self, code): # Change the default identity code.
        self.code[:, :100] = code

    def save(self, save_name):
        pickle.dump({'faces': self.faces, 'vertices': self.vertices, 'identity_weights': self.identity_weights, 'expression_weights': self.expression_weights, 'weights': self.weights, 'identity_names': self.identity_names, 'expression_names': self.expression_names}, open(save_name, 'wb'))

    @classmethod
    # Load the pre-computed weights to accelerate
    def load(self, decimate):
        if decimate == 1:
            load_name = './data_loaders/facs/11089.pkl'
        elif decimate == 2:
            load_name = './data_loaders/facs/3694.pkl'
        else:
            logging.error(f'Only support decimate level in [1, 2], but found {decimate}')
            raise NotImplementedError
        obj = self.__new__(self)
        self.code = np.zeros((1,153))
        obj.faces, obj.vertices, obj.identity_weights, obj.expression_weights, obj.weights, obj.identity_names, obj.expression_names = pickle.load(open(load_name, 'rb'))
        obj.vertices = obj.deform(np.zeros((1, 53)))[0]
        # obj.vertices *= 0.1
        return obj


if __name__ == '__main__':

    ict_model = ICT_model.load(2)
    code = np.zeros((16, 53))
    code[:, 10] = 0.5
    vs = ict_model.deform(code)
    for idx, v in enumerate(vs):
        trimesh.Trimesh(v, ict_model.faces).export(f'./test_{idx:02d}.obj') 


    


