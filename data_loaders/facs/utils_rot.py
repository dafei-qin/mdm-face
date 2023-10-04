import numpy as np

def rot2dof(rot):
    # rot: (F, 3, 3) rotation matrix
    # 6dof: (F, 6) 6dof representation
    
    return rot[:, :, :2].reshape(-1, 6)

def dof2rot(dof):
    # dof: (F, 6) 6dof representation
    # rot: (F, 3, 3) rotation matrix
    
    dof = dof.reshape(-1, 3, 2)
    R0 = dof[:, :, 0] / np.linalg.norm(dof[:, :, 0], axis=1, keepdims=True)

    # R1 = 
    # print(np.einsum('ij,ij->i', R0, R1).shape)
    R1 = dof[:, :, 1] - np.einsum('ij,ij->i', R0, dof[:, :, 1])[..., np.newaxis] * R0
    R1 = R1 / np.linalg.norm(R1, axis=1, keepdims=True)
    R2 = np.cross(R0, R1)
    return np.hstack([R0, R1, R2]).reshape(-1, 3, 3).transpose(0, 2, 1)