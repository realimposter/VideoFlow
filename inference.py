import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.misc import process_cfg
from utils import flow_viz

from core.Networks import build_network

from utils import frame_utils
from VideoFlow.core.utils.utils import InputPadder, forward_interpolate
import itertools
import imageio

def prepare_image(seq_dir):
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    images = []

    image_list = sorted(os.listdir(seq_dir))

    for fn in image_list:
        img = Image.open(os.path.join(seq_dir, fn))
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
    
    return torch.stack(images)

def vis_pre(flow_pre, vis_dir):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
    for idx in range(N//2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

@torch.no_grad()
def MOF_inference(model, cfg):
    start = time.time()
    model.eval()

    image_list = sorted(os.listdir(cfg.seq_dir))

    # Padding: Prepend the first frame and Append the last frame to image_list
    image_list = [image_list[0]] + image_list + [image_list[-1]]

    batch_size = 5
    for i in range(0, len(image_list) - batch_size + 2, batch_size-2):
        print("starting flow batch "+str(i))
        images = []

        start_frame = i
        end_frame = min(i+batch_size, len(image_list))
        batch_image_list = image_list[start_frame:end_frame]
        print(f"start frame: {start_frame}, end frame: {end_frame}")

        for fn in batch_image_list:
            img = Image.open(os.path.join(cfg.seq_dir, fn))
            img = np.array(img).astype(np.uint8)[..., :3]
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            images.append(img)
        
        input_images = torch.stack(images)
        input_images = input_images[None].cuda()
        padder = InputPadder(input_images.shape)
        input_images = padder.pad(input_images)
        flow_pre, _ = model(input_images, {})
        flow_pre = padder.unpad(flow_pre[0]).cpu()
        
        ######### SAVE FLOWS ############
        print("flow_pre shape:", flow_pre.shape)
        N = flow_pre.shape[0]

        # forwards flows
        for idx in range(N//2):
            frame_num = idx + i + 1
            flow_export = flow_pre[idx].permute(1, 2, 0).numpy().astype(np.float16)
            flow = np.zeros((flow_export.shape[0], flow_export.shape[1], 3))
            flow[:, :, 0] = flow_export[:, :, 0]
            flow[:, :, 1] = flow_export[:, :, 1]
            output_path = cfg.output_forward_path.replace("00000", str(frame_num).zfill(5))
            if cfg.compress:
                np.savez_compressed(output_path, flow)
            else: 
                np.save(output_path, flow)

            if cfg.debug:
                viz_path = output_path.replace(".npy", ".png")
                flow_img = flow_viz.flow_to_image(flow_export)
                Image.fromarray(flow_img).save(viz_path)

        # backwards flows
        for idx in range(N//2, N):
            frame_num = idx - N//2 + i
            flow_export = flow_pre[idx].permute(1, 2, 0).numpy().astype(np.float16)
            flow = np.zeros((flow_export.shape[0], flow_export.shape[1], 3))
            flow[:, :, 0] = flow_export[:, :, 0]
            flow[:, :, 1] = flow_export[:, :, 1]
            output_path = cfg.output_backward_path.replace("00000", str(frame_num).zfill(5))
            if cfg.compress:
                np.savez_compressed(output_path, flow)
            else:
                np.save(output_path, flow)
            
            if cfg.debug:
                viz_path = output_path.replace(".npy", ".png")
                flow_img = flow_viz.flow_to_image(flow_export)
                Image.fromarray(flow_img).save(viz_path)
                
    print(f"MOF inference time: {time.time()-start}")
    print("seconds per image: {}".format((time.time()-start)/len(input_images)))
            

@torch.no_grad()
def BOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    
    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    
    with torch.no_grad():
        if args.mode == 'MOF':
            from configs.multiframes_sintel_submission import get_cfg
            flow_pre = MOF_inference(model.module, cfg)
        elif args.mode == 'BOF':
            from configs.sintel_submission import get_cfg
            flow_pre = BOF_inference(model.module, cfg)
    
    vis_pre(flow_pre, cfg.vis_dir)



