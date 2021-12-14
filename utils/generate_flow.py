import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from RAFT.utils.utils import InputPadder

from flow_utils import *

DEVICE = 'cuda'


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask

def run(args, input_path, output_path, output_img_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(input_path, '*.png')) + \
                 glob.glob(os.path.join(input_path, '*.jpg'))

        images = sorted(images)
        for i in range(len(images) - 1):
            print(i)
            image1 = load_image(images[i])
            image2 = load_image(images[i + 1])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_fwd = model(image1, image2, iters=20, test_mode=True)
            _, flow_bwd = model(image2, image1, iters=20, test_mode=True)

            flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
            flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

            mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

            # Save flow
            np.savez(os.path.join(output_path, '%03d_fwd.npz'%i), flow=flow_fwd, mask=mask_fwd)
            np.savez(os.path.join(output_path, '%03d_bwd.npz'%(i + 1)), flow=flow_bwd, mask=mask_bwd)

            # Save flow_img
            Image.fromarray(flow_viz.flow_to_image(flow_fwd)).save(os.path.join(output_img_path, '%03d_fwd.png'%i))
            Image.fromarray(flow_viz.flow_to_image(flow_bwd)).save(os.path.join(output_img_path, '%03d_bwd.png'%(i + 1)))

            Image.fromarray(mask_fwd).save(os.path.join(output_img_path, '%03d_fwd_mask.png'%i))
            Image.fromarray(mask_bwd).save(os.path.join(output_img_path, '%03d_bwd_mask.png'%(i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='Dataset path')
    parser.add_argument('--model', help="restore RAFT checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    input_path = os.path.join(args.dataset_path, 'images')
    output_path = os.path.join(args.dataset_path, 'flow')
    output_img_path = os.path.join(args.dataset_path, 'flow_png')
    create_dir(output_path)
    create_dir(output_img_path)

    run(args, input_path, output_path, output_img_path)
