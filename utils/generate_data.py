import os
import numpy as np
import imageio
import glob
import torch
import torchvision
import skimage.morphology
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def multi_view_multi_time(args):
    """
    Generating multi view multi time data
    """

    Maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    threshold = 0.5

    videoname, ext = os.path.splitext(os.path.basename(args.videopath))

    imgs = []
    reader = imageio.get_reader(args.videopath)
    for i, im in enumerate(reader):
        imgs.append(im)

    imgs = np.array(imgs)
    num_frames, H, W, _ = imgs.shape
    imgs = imgs[::int(np.ceil(num_frames / 100))]

    create_dir(os.path.join(args.data_dir, videoname, 'images'))
    create_dir(os.path.join(args.data_dir, videoname, 'images_colmap'))
    create_dir(os.path.join(args.data_dir, videoname, 'background_mask'))

    for idx, img in enumerate(imgs):
        print(idx)
        imageio.imwrite(os.path.join(args.data_dir, videoname, 'images', str(idx).zfill(3) + '.png'), img)
        imageio.imwrite(os.path.join(args.data_dir, videoname, 'images_colmap', str(idx).zfill(3) + '.jpg'), img)

        # Get coarse background mask
        img = torchvision.transforms.functional.to_tensor(img).to(device)
        background_mask = torch.FloatTensor(H, W).fill_(1.0).to(device)
        objPredictions = Maskrcnn([img])[0]

        for intMask in range(len(objPredictions['masks'])):
            if objPredictions['scores'][intMask].item() > threshold:
                if objPredictions['labels'][intMask].item() == 1: # person
                    background_mask[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

        background_mask_np = ((background_mask.cpu().numpy() > 0.1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(args.data_dir, videoname, 'background_mask', str(idx).zfill(3) + '.jpg.png'), background_mask_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", type=str,
                        help='video path')
    parser.add_argument("--data_dir", type=str, default='../data/',
                        help='where to store data')

    args = parser.parse_args()

    multi_view_multi_time(args)
