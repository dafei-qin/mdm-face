import sys
import os
import trimesh
from glob import glob
import shutil
import ntpath
sys.path.append('/raid/HKU_TK_GROUP/qindafei/motion-diffusion-model/')
sys.path.append('./')
import polyscope as ps
from data_loaders.facs.ICT_model import ICT_model
from data_loaders.facs.facs2obj import mediapipe2ict
from data_loaders.facs.utils_rot import dof2rot
import numpy as np
from copy import deepcopy
from data_loaders.facs.utils_pc2 import writePC2, readPC2
from data_loaders.facs.dataset import facs_data
import subprocess
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
    out_v = out_v / out_v[..., -1:, -1:] # (Frames, V, 4)
    return out_v_bs, out_v[..., :-1]


def save(dataset, all_motions, path, write_names=[], trans_factor=0.1):
    ict_model = init_ict_model(decimate=2)
    face = ict_model.faces
    if not os.path.exists('./ICT_base_mesh.obj'):
        trimesh.Trimesh(ict_model.vertices, face, process=False).export('./ICT_base_mesh.obj')
    if type(all_motions[0]) != np.ndarray:
        all_motions = [d.cpu().numpy() for d in all_motions]
    facs = [d[:52].squeeze().transpose(1, 0) for d in all_motions]
    trans = [d[52:55].squeeze().transpose(1, 0) for d in all_motions]
    rot = [d[55:61].squeeze().transpose(1, 0) for d in all_motions]


    for _idx, (f, t, r) in enumerate(zip(facs, trans, rot)):
        if len(write_names) == 0:
            write_name = f"{_idx:02d}"
        else:
            write_name = write_names[_idx]
        f, t, r = dataset.de_normalize(f, t, r)
        t = t * trans_factor
        r = dof2rot(r, initial_zero=True)
        trans4x4 = np.zeros((r.shape[0], 4, 4))
        trans4x4[:, :3, :3] = r
        trans4x4[:, :3, -1] = t
        trans4x4[:, -1, -1] = 1
        trans4x4[:, :3, -1] -= trans4x4[0, :3, -1]
        assert (np.abs(np.linalg.det(trans4x4[:, :3, :3]) - 1) < 0.01).all()
        vert_seq_bs,  vert_seq= to_mesh(ict_model, f, trans4x4)
        writePC2(path.replace('.npy', f"{write_name}_bs.pc2"), vert_seq_bs, float16=False)
        if t is not None:
            writePC2(path.replace('.npy', f"{write_name}.pc2"), vert_seq, float16=False)

def render_to_video(path, mesh_template, pc2_files, fps=25):
    def render_new_frame(ps_mesh, newPos, filename):
        ps_mesh.update_vertex_positions(newPos)
        ps.screenshot(filename=filename)
    ps.set_screenshot_extension(".jpg")
    ps.init()
    ps.look_at((0, 0, 5), (0, 0, 0))
    mesh = trimesh.load(mesh_template, process=False)
    ps_mesh = ps.register_surface_mesh('biwi', mesh.vertices, mesh.faces)
    # pc2_files = glob(os.path.join(os.path.dirname(path), '*.pc2'))
    for f_name in pc2_files:
        # print(f_name)
        data = readPC2(f_name)['V']
        # print(data.shape)
        for idx in range(len(data)):
            # print(f"{f_name.replace('.pc2', '')}_{idx:04d}.jpg")
            render_new_frame(ps_mesh, data[idx], filename=f"{f_name.replace('.pc2', '')}_{idx:04d}.jpg")

        subprocess.call(['ffmpeg', '-y', '-framerate', str(fps),  '-loglevel', 'quiet', '-i', f"{f_name.replace('.pc2', '')}_%04d.jpg",  f_name.replace('.pc2', '.mp4')])

    for f in glob(os.path.join(path, '*.jpg')):
        os.remove(f)
        
        


if __name__ == '__main__':
    import pickle
    from data_loaders.facs.facs2obj import expression_bases_ICT, expression_bases_mediapipe, mapping, mapping_list
    # dataset = facs_data(datapath='/data0/qindafei/celebv-text/processed', num_frames=-1)
    # dataset = facs_data(datapath='/data0/qindafei/celebv-text/pkl_25fps', num_frames=-1)
    dataset = facs_data(datapath='/data0/qindafei/MEAD_pkl', mead=True, num_frames=-1)
    data = [dataset[i] for i in np.arange(10)]
    all_motions = [d['inp'] for d in data]
    fps = 30
    # out_dir = './debug'
    # out_dir = './debug_25fps'
    out_dir = './debug_mead'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Create dir: ', out_dir)
    save(dataset, all_motions, path=f'./{out_dir}/out.npy')
    pc2_files = glob(f'./{out_dir}/*.pc2')
    pc2_files = [f for f in pc2_files if 'bs' not in f]
    render_to_video(out_dir, './ICT_base_mesh.obj', pc2_files, fps=fps)
    files = sorted(glob('/data0/qindafei/celebv-text/pkl/*.pkl'))
    videos = sorted(glob(f'./{out_dir}/*.mp4'))
    videos = [v for v in videos if '_bs' not in v and 'sub' not in v]
    for idx, d in enumerate(data):
        name = ntpath.basename(d['name'])
        action = os.path.join('/data0/qindafei/celebv-text/action_dur/', name + '.txt')
        try:
            action = open(action, 'r').readline()
        except FileNotFoundError:
            print(name, 'Corresponding text file not exist! Only rename .mp4')
            shutil.move(videos[idx], os.path.join(out_dir, f'{name}.mp4'))
            continue
        action = action.split(' ')
        action = ''.join([action[idx]+' ' if idx % 20 else '\n' for idx in range(len(action))])
        # action += '\n'
        emotion = os.path.join('/data0/qindafei/celebv-text/emotion/', name + '.txt')
        emotion = open(emotion, 'r').readline()
        emotion = emotion.replace('\n', '').split()
        emotion = ''.join([emotion[idx] + ' ' if idx % 20 else '\n' for idx in range(len(emotion))])
        emotion += '\n'
        vid = videos[idx]
        subprocess.call(['ffmpeg', '-y',  '-loglevel', 'quiet', '-i', vid, "-vf", f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{action+emotion}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=(h-text_h)", os.path.join(out_dir, f'{name}.mp4')])
    # facs, mean, std = data.values()
    # facs = np.concatenate(facs, axis=0)
    # facs = facs * std + mean
    # facs_ICT = facs[..., mapping_list]
    print()