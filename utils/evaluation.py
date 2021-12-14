import os
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity


def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def readimage(data_dir, sequence, time, method):
    img = cv2.imread(os.path.join(data_dir, method, sequence, 'v000_t' + str(time).zfill(3) + '.png'))
    return img


def calculate_metrics(data_dir, sequence, methods, lpips_loss):

    PSNRs = np.zeros((len(methods)))
    SSIMs = np.zeros((len(methods)))
    LPIPSs = np.zeros((len(methods)))

    nFrame = 0

    # Yoon's results do not include v000_t000 and v000_t011. Omit these two
    # frames if evaluating Yoon's method.
    if 'Yoon' in methods:
        time_start = 1
        time_end = 11
    else:
        time_start = 0
        time_end = 12

    for time in range(time_start, time_end): # Fix view v0, change time

        nFrame += 1

        img_true = readimage(data_dir, sequence, time, 'gt')

        for method_idx, method in enumerate(methods):

            if 'Yoon' in methods and sequence == 'Truck' and time == 10:
                break

            img = readimage(data_dir, sequence, time, method)
            PSNR = cv2.PSNR(img_true, img)
            SSIM = structural_similarity(img_true, img, multichannel=True)
            LPIPS = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()

            PSNRs[method_idx] += PSNR
            SSIMs[method_idx] += SSIM
            LPIPSs[method_idx] += LPIPS

    PSNRs = PSNRs / nFrame
    SSIMs = SSIMs / nFrame
    LPIPSs = LPIPSs / nFrame

    return PSNRs, SSIMs, LPIPSs


if __name__ == '__main__':

    lpips_loss = lpips.LPIPS(net='alex') # best forward scores
    data_dir = '../results'
    sequences = ['Balloon1', 'Balloon2', 'Jumping', 'Playground', 'Skating', 'Truck', 'Umbrella']
    # methods = ['NeRF', 'NeRF_t', 'Yoon', 'NR', 'NSFF', 'Ours']
    methods = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours']

    PSNRs_total = np.zeros((len(methods)))
    SSIMs_total = np.zeros((len(methods)))
    LPIPSs_total = np.zeros((len(methods)))
    for sequence in sequences:
        print(sequence)
        PSNRs, SSIMs, LPIPSs = calculate_metrics(data_dir, sequence, methods, lpips_loss)
        for method_idx, method in enumerate(methods):
            print(method.ljust(7) + '%.2f'%(PSNRs[method_idx]) + ' / %.4f'%(SSIMs[method_idx]) + ' / %.3f'%(LPIPSs[method_idx]))

        PSNRs_total += PSNRs
        SSIMs_total += SSIMs
        LPIPSs_total += LPIPSs

    PSNRs_total = PSNRs_total / len(sequences)
    SSIMs_total = SSIMs_total / len(sequences)
    LPIPSs_total = LPIPSs_total / len(sequences)
    print('Avg.')
    for method_idx, method in enumerate(methods):
        print(method.ljust(7) + '%.2f'%(PSNRs_total[method_idx]) + ' / %.4f'%(SSIMs_total[method_idx]) + ' / %.3f'%(LPIPSs_total[method_idx]))
