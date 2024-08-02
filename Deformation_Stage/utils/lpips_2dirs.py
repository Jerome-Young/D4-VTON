import argparse
import os
import lpips
import torch
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input1', type=str, default='./imgs/ex_dir1')
parser.add_argument('--input2', type=str, default='./imgs/ex_dir2')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
device = torch.device("cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu")
loss_fn.to(device)

# crawl directories
files = os.listdir(opt.input1)
LPIPS = []
for file in files:
    if os.path.exists(os.path.join(opt.input2, file)):
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.input1, file)))
        img2 = lpips.im2tensor(lpips.load_image(os.path.join(opt.input2, file)))
        img1 = img1.to(device)
        img2 = img2.to(device)
        dist01 = loss_fn.forward(img1, img2)
        LPIPS.append(dist01.item())
mean_LPIPS = torch.mean(torch.tensor(LPIPS))
print("Mean LPIPS:", np.array(mean_LPIPS))

