import cv2
import glob
import os
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import argparse
from PIL import Image
SSIMS =[]
PSNRS = []
import torch
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input1',  type=str)
parser.add_argument('--input2',  type=str)
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
opt = parser.parse_args()
#ssim_loss = pytorch_ssim.SSIM(window_size = 11)
img2_files = glob.glob(opt.input2+"/*")
device = torch.device("cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu")
for file2 in img2_files:
    # Extract the filename from the path
    fake_filename = os.path.basename(file2)
    # Find the corresponding real image file
    real_file = os.path.join(opt.input1, fake_filename)

    if os.path.exists(real_file):
        real = cv2.imread(real_file)
        fake = cv2.imread(file2)

        if real.shape[0] != fake.shape[0] or real.shape[1] != fake.shape[1]:
            pil_img = Image.fromarray(fake)
            pil_img = pil_img.resize((real.shape[1], real.shape[0]))
            hazy_img = np.array(pil_img)

        # 计算PSNR
        PSNR = peak_signal_noise_ratio(real, fake)
        PSNRS.append(PSNR)

        # 计算SSIM
        SSIM = structural_similarity(real, fake, multichannel=True)
        SSIMS.append(SSIM)

print('PSNR: ', sum(PSNRS) / len(PSNRS))
print('SSIM: ', sum(SSIMS) / len(SSIMS))
