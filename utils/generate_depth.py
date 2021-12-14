"""Compute depth maps for images in the input folder.
"""
import os
import cv2
import glob
import torch
import argparse
import numpy as np

from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def run(input_path, output_path, output_img_path, model_path):
    """Run MonoDepthNN to compute depth maps.
    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)
    sh = cv2.imread(sorted(glob.glob(os.path.join(input_path, "*")))[0]).shape
    net_w, net_h = sh[1], sh[0]

    resize_mode="upper_bound"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.eval()
    model.to(device)

    # get input
    img_names = sorted(glob.glob(os.path.join(input_path, "*")))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=[net_h, net_w],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )

        print(filename + '.npy')
        np.save(filename + '.npy', prediction.astype(np.float32))

        depth_min = prediction.min()
        depth_max = prediction.max()

        max_val = (2**(8*2))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (prediction - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(prediction.shape, dtype=prediction.type)

        cv2.imwrite(os.path.join(output_img_path, os.path.splitext(os.path.basename(img_name))[0] + '.png'), out.astype("uint16"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='Dataset path')
    parser.add_argument('--model', help="restore midas checkpoint")
    args = parser.parse_args()

    input_path = os.path.join(args.dataset_path, 'images')
    output_path = os.path.join(args.dataset_path, 'disp')
    output_img_path = os.path.join(args.dataset_path, 'disp_png')
    create_dir(output_path)
    create_dir(output_img_path)

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(input_path, output_path, output_img_path, args.model)
