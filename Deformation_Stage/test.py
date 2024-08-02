import argparse
import os
import torch.backends.cudnn as cudnn
import torch
from torch.nn import functional as F
import tqdm
import numpy as np
from data.aligned_dataset_vitonhd import AlignedDataset
from models.dsdnet import DSDNet
from torch.utils import data
from torchvision.utils import save_image
cudnn.benchmark = True


def load_networks(opt, network, load_path):
    device = torch.device("cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu")
    if not os.path.exists(load_path):
        print("not exsits %s" % load_path)
        return
    print('loading the model from %s' % load_path)

    state_dict = torch.load(load_path, map_location=device)
    # load params
    network.load_state_dict(state_dict["state_dict"])

    return network


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--load_height', type=int, default=512)
    parser.add_argument('--load_width', type=int, default=384)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--sample_nums', type=int, default=1)
    parser.add_argument('--group_nums', type=int, default=8)
    # dataset
    parser.add_argument('--dataroot', type=str, default='/VITON-HD-512/')
    parser.add_argument('--image_pairs_txt', type=str, default='test_pairs_unpaired_1018.txt')
    parser.add_argument('--label_nc', type=int, default=14, help='# of input label channels')
    parser.add_argument('--resize_or_crop', type=str, default='None',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
    parser.add_argument('--no_flip', action='store_true',
                             help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
    parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
    # test setting
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--exp_name', type=str, default='unpaired-cloth-warp')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/vitonhd_deformation.pt')
    opt = parser.parse_args()
    return opt


def deformation_test(opt, warp_model):
    test_dataset = AlignedDataset()
    test_dataset.initialize(opt)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    with torch.no_grad():
        for i, inputs in enumerate(tqdm.tqdm(test_loader)):
            img_names = inputs['img_path']
            pre_clothes_edge = inputs['edge']
            clothes = inputs['color']
            clothes = clothes * pre_clothes_edge
            real_image = inputs['image']
            pose = inputs['pose']
            size = inputs['color'].size()
            oneHot_size1 = (size[0], 25, size[2], size[3])
            densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
            densepose = densepose.scatter_(1, inputs['densepose'].data.long().cuda(), 1.0)
            densepose = densepose * 2.0 - 1.0

            pose = pose.cuda()
            clothes = clothes.cuda()
            preserve_mask = inputs['preserve_mask'].cuda()

            condition = torch.cat([densepose, pose, preserve_mask], 1)
            results_all = warp_model(condition, clothes)

            for j in range(real_image.shape[0]):
                save_image(results_all[-1][j:j+1], os.path.join(opt.save_dir, opt.name, opt.exp_name, img_names[j].split('/')[-1]),
                           nrow=1, normalize=True, range=(-1, 1))


def main():
    opt = get_opt()
    print(opt)
    torch.cuda.set_device("cuda:" + str(opt.gpu))
    if not os.path.exists(os.path.join(opt.save_dir, opt.name, opt.exp_name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name, opt.exp_name))

    # define model
    warp_model = DSDNet(cond_in_channel=51, sample_nums=opt.sample_nums, group_nums=opt.group_nums).cuda()
    warp_model.eval()
    load_networks(opt, warp_model, opt.ckpt_dir)

    deformation_test(opt, warp_model)


if __name__ == '__main__':
    main()

