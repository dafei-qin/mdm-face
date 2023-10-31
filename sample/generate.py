# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate, verts_collate
from data_loaders.facs.dataset import facs_data
import soundfile as sf
from copy import deepcopy
import trimesh
from data_loaders.biwi.render import init_renderer, render_sequence
def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = args.num_frames
    # max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    if args.dataset == 'biwi':
        fps = 25
    n_frames = max_frames
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name, args.audio_file, args.inpainting, args.dataset == 'facs'])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
    write_names = []
    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)
    elif args.mead_file != '':
        assert os.path.exists(args.mead_file)
        with open(args.mead_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [a.strip() for a in action_text]
        # action_text = [s.split(' ') for s in action_text]
        args.num_samples = len(action_text)
    elif args.dataset == 'facs':
        if args.inpainting_file != '':
            assert os.path.exists(args.inpainting_file)
            files = open(args.inpainting_file, 'r').readlines()
            files = [s.replace('\n', '') for s in files]
            files = [s for s in files if s != '']
            files = [(int(s.split('_')[0]), float(s.split('_')[1])) for s in files]
        else:
            files = [(0, 0.1), (5, 0.2), (10, 0.3), (15, 0.4)] # Just chose some random sequences
        args.num_samples = len(files)
    else:
        args.num_samples = 1

    if args.audio_file != '':
        assert os.path.exists(args.audio_file)
        audio_files = open(args.audio_file, 'r').readlines()
        audio_files = [s.replace('\n', '') for s in audio_files]
        audio_files = [s for s in audio_files if s != '']
        args.num_samples *= len(audio_files)
    
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        elif args.action_file != '':
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text, name=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        if args.audio_file != '':
            import librosa
            from transformers import Wav2Vec2Processor
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            wavs = [librosa.load(f, sr=16000)[0] for f in audio_files]
            wavs_name = [os.path.basename(f).split('.')[0] for f in audio_files]
            fps2ar = 16000 / 25
            if args.action_file != '': # Iterate though all identities and audios.
                collate_args = [deepcopy(arg) for arg in collate_args for idx in range(len(wavs))]
                wavs = [deepcopy(wavs[i]) for k in range(len(action)) for i in range(len(wavs_name))]
                for idx in range(len(collate_args)):
                     collate_args[idx]['name'] = f'{action_text[idx // len(wavs_name)]}_{wavs_name[idx % len(wavs_name)]}'
            write_names = [deepcopy(arg['name']) for arg in collate_args]
            def trim_audios(processor, collate_args, wavs, max_ar):
                max_ar = int(max_ar)
                collate_args = [dict(arg, idx=idx_input) for idx_input, arg in enumerate(collate_args)]

                wavs_trims = []
                for idx, wav in enumerate(wavs):
                    
                    while len(wav) > max_ar:
                        wav_trim = wav[:max_ar]
                        wavs_trims.append((wav_trim, idx))
                        wav = wav[max_ar:]
                    wav_trim = np.zeros(max_ar)
                    wav_trim[:len(wav)] = wav
                    wavs_trims.append((wav_trim, idx))
                collate_args = [dict(collate_args[idx], au=torch.from_numpy(processor(wav_trim, sampling_rate=16000).input_values[0]).to(dist_util.dev())) for (wav_trim, idx) in wavs_trims]
                return collate_args
            collate_args = trim_audios(processor, collate_args, wavs, max_frames * fps2ar)


        args.num_samples = len(collate_args)
        args.batch_size = args.num_samples

        if args.mead_file != '':
            action = data.dataset.action_name_to_action([a.split(' ')[0] for a in action_text])
            var = [float(a.split(' ')[1]) for a in action_text] 
            collate_args = [dict(arg, action=one_action, action_text=one_action_text, name=one_action_text, var=one_var) for
                            arg, one_action, one_action_text, one_var in zip(collate_args, action, action_text, var)]
        elif args.dataset == 'facs':
            inpainting = args.inpainting
            if args.data_dir == '':
                dataset = facs_data(split='test', num_frames=args.num_frames, inpainting=inpainting, sampling='disabled') # Disable data sampling 
            else:
                dataset = facs_data(datapath=args.data_dir, split='test', num_frames=args.num_frames, inpainting=inpainting, sampling='disabled')

            collate_args = [dataset[i[0]] for i in files]
            orig_collate_args = deepcopy(collate_args)
            for idx, _c in enumerate(collate_args):
                _c['var'] = files[idx][1]
                _c['action_text'] = f'{files[idx][1]:.2f}'
            
        if args.mead_file:
            write_names = [a.replace(' ', '_') + f'_{i}' for i in range(args.num_repetitions) for a in action_text]
        elif args.cond_var:
            write_names = [f'_idx={f[0]:03d}_std={f[1]:.2f}' for f in files]
            write_names = [f'{w}_rep={r:02d}' for r in range(args.num_repetitions) for w in write_names]
        if args.dataset in ['biwi', 'facs']:
            gts, model_kwargs = verts_collate(collate_args)

        else:
            gts, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []
    all_audios = []
    model_kwargs_bk = deepcopy(model_kwargs)
    for rep_i in range(args.num_repetitions):
        model_kwargs = deepcopy(model_kwargs_bk)
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        # Here add the beginning and the ending frames to do inpainting
        init_image = None
        if args.inpainting:
            init_image = gts
            init_image = init_image.to(dist_util.dev())
            model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].to(dist_util.dev())
            model_kwargs['y']['inpainted_motion'] = model_kwargs['y']['inpainted_motion'].to(dist_util.dev())
            
        if args.double_take:
            from utils.double_take_utils import double_take_arb_len, unfold_sample_arb_len

            samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, model_kwargs, n_frames,  eval_mode=False)
            step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
            for ii, len_i in enumerate(model_kwargs['y']['lengths']):
                if ii == 0:
                    step_sizes[ii] = len_i
                    continue
                step_sizes[ii] = step_sizes[ii-1] + len_i - args.handshake_size

            final_n_frames = step_sizes[-1]

            for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

                sample = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)

                all_motions.append(sample.squeeze().cpu().numpy()[np.newaxis])
                all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        else:     
            sample = sample_fn(
                model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=init_image,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            if args.inpainting:
                sample[..., 0] = init_image[..., 0]
                sample[..., -1] = init_image[..., -1]
            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            audios = []
            if args.audio_file != '':
                for idx in range(len(wavs)):
                    
                    audios.append([])
                    for _idx, arg in enumerate(collate_args):
                        if arg['idx'] == idx:
                            audios[idx].append(all_motions[-1][_idx])
                            
                audios = [np.concatenate(i, axis=-1) for i in audios]
                audios = [audio[..., :int(np.floor(len(wavs[_idx]) / fps2ar))] for _idx, audio in enumerate(audios)] # Clip to the length of the audio
            for au in audios:
                all_audios.append(au)
            print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    if model.data_rep == 'face_verts':
        from data_loaders.biwi.render import save
        if args.audio_file != '':
            all_motions = all_audios
            if args.render:
                mesh = trimesh.load('./biwi_ssp_8000_deformed.obj', process=False)
                ps_mesh = init_renderer(mesh)
                for idx in range(len(all_motions)):
                    seq = all_motions[idx]
                    wav = wavs[idx]
                    filename = write_names[idx]
                    print(seq.shape, wav.shape, os.path.join(os.path.dirname(npy_path), filename + '.mp4'))
                    render_sequence(ps_mesh, seq.transpose(2, 0, 1), os.path.join(os.path.dirname(npy_path), filename+'.mp4', ), audio=wav)
            else:
                save(all_motions, write_names, npy_path, wavs)
        else:
            save(all_motions, write_names, npy_path)
        print(f'PC2 results at at [{os.path.abspath(os.path.dirname(npy_path))}]')
    elif model.data_rep == 'facs':
        from data_loaders.facs.render import save
        save(data.dataset, all_motions, npy_path, write_names[:len(all_motions)])
        try:
            gts_name = [f"gt_{i[0]:02d}_std={k['var'].item():.2f}" for (i, k) in zip(files, orig_collate_args)]
            save(data.dataset, gts, npy_path, gts_name)
        except:
            print('GT saving failed, pass')
    else:
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        sample_files = []
        num_samples_in_out_file = 7

        sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

        for sample_i in range(args.num_samples):
            rep_files = []
            for rep_i in range(args.num_repetitions):
                caption = all_text[rep_i*args.batch_size + sample_i]
                length = all_lengths[rep_i*args.batch_size + sample_i]
                motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                save_file = sample_file_template.format(sample_i, rep_i)
                print(sample_print_template.format(caption, sample_i, rep_i, save_file))
                animation_save_path = os.path.join(out_path, save_file)
                plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
                rep_files.append(animation_save_path)

            sample_files = save_multiple_samples(args, out_path,
                                                row_print_template, all_print_template, row_file_template, all_file_template,
                                                caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only', data_dir=args.data_dir)
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
