import os
import cv2
import PIL
import glob
import torch
import argparse
import numpy as np

from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

import skimage.morphology
import torchvision
from flow_utils import read_optical_flow, compute_epipolar_distance, skew



def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_poses(im):
    R = im.qvec2rotmat()
    t = im.tvec.reshape([3,1])
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

    return m


def load_colmap_data(realdir):

    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)

    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)

    w2c_mats = []
    # bottom = np.array([0,0,0,1.]).reshape([1,4])

    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print( 'Images #', len(names))
    perm = np.argsort(names)

    return imdata, perm, img_keys, hwf


def run_maskrcnn(model, img_path, intWidth=1024, intHeight=576):

    # intHeight = 576
    # intWidth = 1024

    threshold = 0.5

    o_image = PIL.Image.open(img_path)
    image = o_image.resize((intWidth, intHeight), PIL.Image.ANTIALIAS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions['masks'].size(0)):
        if objPredictions['scores'][intMask].item() > threshold:
            if objPredictions['labels'][intMask].item() == 1: # person
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 4: # motorcycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 2: # bicycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 8: # truck
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 28: # umbrella
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 17: # cat
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 18: # dog
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 36: # snowboard
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 41: # skateboard
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(tenHumans.cpu().numpy(),
                                         skimage.morphology.disk(1))
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask


def motion_segmentation(basedir, threshold,
                        input_semantic_w=1024,
                        input_semantic_h=576):

    points3dfile = os.path.join(basedir, 'sparse/0/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)

    img_dir = glob.glob(basedir + '/images_colmap')[0]
    img0 = glob.glob(glob.glob(img_dir)[0] + '/*jpg')[0]
    shape_0 = cv2.imread(img0).shape

    resized_height, resized_width = shape_0[0], shape_0[1]

    imdata, perm, img_keys, hwf = load_colmap_data(basedir)
    scale_x, scale_y = resized_width / float(hwf[1]), resized_height / float(hwf[0])

    K = np.eye(3)
    K[0, 0] = hwf[2]
    K[0, 2] = hwf[1] / 2.
    K[1, 1] = hwf[2]
    K[1, 2] = hwf[0] / 2.

    xx = range(0, resized_width)
    yy = range(0, resized_height)
    xv, yv = np.meshgrid(xx, yy)
    p_ref = np.float32(np.stack((xv, yv), axis=-1))
    p_ref_h = np.reshape(p_ref, (-1, 2))
    p_ref_h = np.concatenate((p_ref_h, np.ones((p_ref_h.shape[0], 1))), axis=-1).T

    num_frames = len(perm)

    if os.path.isdir(os.path.join(basedir, 'images_colmap')):
        num_colmap_frames = len(glob.glob(os.path.join(basedir, 'images_colmap', '*.jpg')))
        num_data_frames = len(glob.glob(os.path.join(basedir, 'images', '*.png')))

        if num_colmap_frames != num_data_frames:
            num_frames = num_data_frames


    save_mask_dir = os.path.join(basedir, 'motion_segmentation')
    create_dir(save_mask_dir)

    for i in range(0, num_frames):
        im_prev = imdata[img_keys[perm[max(0, i - 1)]]]
        im_ref = imdata[img_keys[perm[i]]]
        im_post = imdata[img_keys[perm[min(num_frames -1, i + 1)]]]

        print(im_prev.name, im_ref.name, im_post.name)

        T_prev_G = extract_poses(im_prev)
        T_ref_G = extract_poses(im_ref)
        T_post_G = extract_poses(im_post)

        T_ref2prev = np.dot(T_prev_G, np.linalg.inv(T_ref_G))
        T_ref2post = np.dot(T_post_G, np.linalg.inv(T_ref_G))
        # load optical flow

        if i == 0:
          fwd_flow, _ = read_optical_flow(basedir,
                                       im_ref.name,
                                       read_fwd=True)
          bwd_flow = np.zeros_like(fwd_flow)
        elif i == num_frames - 1:
          bwd_flow, _ = read_optical_flow(basedir,
                                       im_ref.name,
                                       read_fwd=False)
          fwd_flow = np.zeros_like(bwd_flow)
        else:
          fwd_flow, _ = read_optical_flow(basedir,
                                       im_ref.name,
                                       read_fwd=True)
          bwd_flow, _ = read_optical_flow(basedir,
                                       im_ref.name,
                                       read_fwd=False)

        p_post = p_ref + fwd_flow
        p_post_h = np.reshape(p_post, (-1, 2))
        p_post_h = np.concatenate((p_post_h, np.ones((p_post_h.shape[0], 1))), axis=-1).T

        fwd_e_dist = compute_epipolar_distance(T_ref2post, K,
                                               p_ref_h, p_post_h)
        fwd_e_dist = np.reshape(fwd_e_dist, (fwd_flow.shape[0], fwd_flow.shape[1]))

        p_prev = p_ref + bwd_flow
        p_prev_h = np.reshape(p_prev, (-1, 2))
        p_prev_h = np.concatenate((p_prev_h, np.ones((p_prev_h.shape[0], 1))), axis=-1).T

        bwd_e_dist = compute_epipolar_distance(T_ref2prev, K,
                                               p_ref_h, p_prev_h)
        bwd_e_dist = np.reshape(bwd_e_dist, (bwd_flow.shape[0], bwd_flow.shape[1]))

        e_dist = np.maximum(bwd_e_dist, fwd_e_dist)

        motion_mask = skimage.morphology.binary_opening(e_dist > threshold, skimage.morphology.disk(1))

        cv2.imwrite(os.path.join(save_mask_dir, im_ref.name.replace('.jpg', '.png')), np.uint8(255 * (0. + motion_mask)))

    # RUN SEMANTIC SEGMENTATION
    img_dir = os.path.join(basedir, 'images')
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                  + sorted(glob.glob(os.path.join(img_dir, '*.png')))
    semantic_mask_dir = os.path.join(basedir, 'semantic_mask')
    netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    create_dir(semantic_mask_dir)


    for i in range(0, len(img_path_list)):
        img_path = img_path_list[i]
        img_name = img_path.split('/')[-1]
        semantic_mask = run_maskrcnn(netMaskrcnn, img_path,
                                     input_semantic_w,
                                     input_semantic_h)
        cv2.imwrite(os.path.join(semantic_mask_dir,
                                img_name.replace('.jpg', '.png')),
                    semantic_mask)

    # combine them
    save_mask_dir = os.path.join(basedir, 'motion_masks')
    create_dir(save_mask_dir)

    mask_dir = os.path.join(basedir, 'motion_segmentation')
    mask_path_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    semantic_dir = os.path.join(basedir, 'semantic_mask')

    for mask_path in mask_path_list:
        print(mask_path)

        motion_mask = cv2.imread(mask_path)
        motion_mask = cv2.resize(motion_mask, (resized_width, resized_height),
                                interpolation=cv2.INTER_NEAREST)
        motion_mask = motion_mask[:, :, 0] > 0.1

        # combine from motion segmentation
        semantic_mask = cv2.imread(os.path.join(semantic_dir, mask_path.split('/')[-1]))
        semantic_mask = cv2.resize(semantic_mask, (resized_width, resized_height),
                                interpolation=cv2.INTER_NEAREST)
        semantic_mask = semantic_mask[:, :, 0] > 0.1
        motion_mask = semantic_mask | motion_mask

        motion_mask = skimage.morphology.dilation(motion_mask, skimage.morphology.disk(2))
        cv2.imwrite(os.path.join(save_mask_dir, '%s'%mask_path.split('/')[-1]),
                    np.uint8(np.clip((motion_mask), 0, 1) * 255) )

    # delete old mask dir
    os.system('rm -r %s'%mask_dir)
    os.system('rm -r %s'%semantic_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='Dataset path')
    parser.add_argument("--epi_threshold", type=float,
                        default=1.0,
                        help='epipolar distance threshold for physical motion segmentation')

    parser.add_argument("--input_flow_w", type=int,
                        default=768,
                        help='input image width for optical flow, \
                        the height will be computed based on original aspect ratio ')

    parser.add_argument("--input_semantic_w", type=int,
                        default=1024,
                        help='input image width for semantic segmentation')

    parser.add_argument("--input_semantic_h", type=int,
                        default=576,
                        help='input image height for semantic segmentation')
    args = parser.parse_args()

    motion_segmentation(args.dataset_path, args.epi_threshold,
                        args.input_semantic_w,
                        args.input_semantic_h)
