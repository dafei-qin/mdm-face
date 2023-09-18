import trimesh
import polyscope as ps
import numpy as np
import subprocess
import os
import soundfile as sf
def init_renderer(mesh):
    ps.set_screenshot_extension('.jpg')
    ps.init()
    ps.look_at((0, 0, 3), (0, 0, 0))
    ps_mesh = ps.register_surface_mesh('biwi', mesh.vertices, mesh.faces)
    return ps_mesh


def render_new_frame(ps_mesh, newPos, filename):
    ps_mesh.update_vertex_positions(newPos)
    ps.screenshot(filename=filename)
    


def render_sequence(ps_mesh, newPoss, filename, audio=None):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    tmp_dir = f'./tmp/{os.path.basename(filename)}'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for idx, pos in enumerate(newPoss):
        render_new_frame(ps_mesh, pos, os.path.join(tmp_dir, f'{idx:04d}.jpg'))
    if audio is not None:
        sf.write(filename.replace('.mp4', '.wav'), audio, 16000)
        subprocess.call(['ffmpeg', '-y', '-loglevel', 'quiet',  '-i', f'{tmp_dir}/%04d.jpg', '-i', filename.replace('.mp4', '.wav'), filename])
    else:
        subprocess.call(['ffmpeg', '-y', '-loglevel', 'quiet',  '-i', f'{tmp_dir}/%04d.jpg', filename])

def save(all_motions, write_names, path, wavs=None):
    from data_loaders.facs.utils_pc2 import writePC2
    for idx, motion in enumerate(all_motions):

        writePC2(path.replace('.npy', f"_rep_{idx // len(write_names)}_{write_names[idx % len(write_names)]}.pc2"), motion.transpose(2, 0, 1), float16=False)


        if wavs is not None:
            sf.write(path.replace('.npy', f"_rep_{idx // len(write_names)}_{write_names[idx % len(write_names)]}.wav"), wavs[idx % len(write_names)], 16000)

if __name__ == '__main__':
    ps.set_screenshot_extension(".jpg")
    ps.init()
    ps.look_at((0, 0, 3), (0, 0, 0))
    mesh = trimesh.load('../../biwi_8000_neutral.obj', process=False)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    ps_mesh = ps.register_surface_mesh('biwi', mesh.vertices, mesh.faces)
    ps.screenshot()

    mesh_diff = trimesh.load('../../biwi_ssp_8000_deformed.obj', process=False)
    mesh_diff.vertices -= mesh_diff.vertices.mean(axis=0)
    ps_mesh.update_vertex_positions(mesh_diff.vertices)
    ps.screenshot()