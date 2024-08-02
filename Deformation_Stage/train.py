from models.dsdnet import DSDNet
import os
import torch, argparse, wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm
from models import external_function
from utils import lpips
from utils.utils import AverageMeter
from torchvision import utils
from data.aligned_dataset_vitonhd import AlignedDataset


def load_last_checkpoint(opt, network, optimizer):
    load_path = opt.save_dir + opt.name + f"/{str(opt.continue_from_epoch).zfill(3)}_viton_{str(opt.name)}.pt"
    if not os.path.exists(load_path):
        print("not exsits %s" % load_path)
        return
    print('loading the model from %s' % load_path)

    checkpoint = torch.load(load_path, map_location='cuda:{}'.format(opt.local_rank))
    network.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optim"])


def load_checkpoint_parallel(opt, network, load_path):
    if not os.path.exists(load_path):
        print("not exsits %s" % load_path)
        return
    print('loading the model from %s' % load_path)
    checkpoint = torch.load(load_path, map_location='cuda:{}'.format(opt.local_rank))
    checkpoint_new = network.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint['state_dict'][param]
    network.load_state_dict(checkpoint_new)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=512)
    parser.add_argument('--load_width', type=int, default=384)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--sample_nums', type=int, default=1)
    parser.add_argument('--group_nums', type=int, default=8)
    # dataset
    parser.add_argument('--dataroot', type=str, default='/VITON-HD-512/')
    parser.add_argument('--image_pairs_txt', type=str, default='train_pairs_1018.txt')
    parser.add_argument('--label_nc', type=int, default=14, help='# of input label channels')
    parser.add_argument('--resize_or_crop', type=str, default='None',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
    parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
    # log & checkpoints
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--display_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parser.add_argument('--continue_from_epoch', default=0, type=int, help='Continue training from epoch (default=0)')
    parser.add_argument('--epochs', default=80, type=int, help='training epochs (default=80)')
    parser.add_argument('--light_dir', type=str, default='./checkpoints/hd_lightnet.pt')

    opt = parser.parse_args()
    return opt


def train(opt):
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{opt.local_rank}')

    train_data = AlignedDataset()
    train_data.initialize(opt)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, sampler=train_sampler)
    dataset_size = len(train_loader)

    warp_model = DSDNet(cond_in_channel=51, sample_nums=opt.sample_nums, group_nums=opt.group_nums).cuda()
    warp_model.train()

    optimizer = torch.optim.AdamW(warp_model.parameters(), lr=opt.lr)

    if opt.continue_from_epoch > 0:
        load_last_checkpoint(opt, warp_model, optimizer)
    else:
        load_checkpoint_parallel(opt, warp_model.dsdms.lightnet, opt.light_dir)
    warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

    if opt.num_gpus != 0:
        warp_model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.local_rank])

    # criterion
    criterion_L1 = nn.L1Loss()
    criterion_percept = lpips.exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=False)
    criterion_style = external_function.VGGLoss().to(device)

    if opt.local_rank == 0:
        wandb.init(project="d4-vton", name=opt.name, settings=wandb.Settings(code_dir="."))
        print('#training images = %d' % dataset_size)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2,
                                                last_epoch=opt.continue_from_epoch - 1)

    for epoch in range(opt.continue_from_epoch, opt.epochs):
        loss_l1_avg = AverageMeter()
        loss_vgg_avg = AverageMeter()
        train_sampler.set_epoch(epoch)
        iterations = 0

        for data in tqdm.tqdm(train_loader):
            iterations += 1
            pre_clothes_edge = data['edge']
            clothes = data['color']
            clothes = clothes * pre_clothes_edge
            person_clothes_edge = data['person_clothes_mask']
            real_image = data['image']
            person_clothes = real_image * person_clothes_edge
            pose = data['pose']
            size = data['color'].size()
            oneHot_size1 = (size[0], 25, size[2], size[3])
            densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
            densepose = densepose.scatter_(1, data['densepose'].data.long().cuda(), 1.0)
            densepose = densepose * 2.0 - 1.0

            person_clothes = person_clothes.cuda()
            pose = pose.cuda()
            clothes = clothes.cuda()
            preserve_mask = data['preserve_mask'].cuda()

            condition = torch.cat([densepose, pose, preserve_mask], 1)
            results_all = warp_model(condition, clothes)

            loss_all = 0
            num_layer = 5
            for num in range(num_layer):
                if num == 1 or num == 3:
                  continue
                cur_img = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
                loss_l1 = criterion_L1(results_all[num], cur_img)
                if num == 0:
                    cur_img = F.interpolate(cur_img, scale_factor=2, mode='bilinear')
                    results_all[num] = F.interpolate(results_all[num], scale_factor=2, mode='bilinear')
                loss_perceptual = criterion_percept(cur_img, results_all[num]).mean()
                loss_content, loss_style = criterion_style(results_all[num], cur_img)
                loss_vgg = loss_perceptual + 100 * loss_style + 0.1 * loss_content
                loss_all = loss_all + (num + 1) * loss_l1 + (num + 1) * loss_vgg

            loss = loss_all
            loss_l1_avg.update(loss_all.item())
            loss_vgg_avg.update(loss_vgg.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % 50 == 1 and opt.local_rank == 0:
                wandb.log({'l1_loss': loss_l1_avg.avg,
                           'vgg_loss': loss_vgg_avg.avg,
                           'epoch': epoch, 'steps': iterations})

            if iterations % opt.display_freq == 0 and opt.local_rank == 0:
                parse_pred = torch.cat([real_image.cuda(), clothes, results_all[-1], person_clothes], 3)
                utils.save_image(
                    parse_pred,
                    f"{os.path.join(opt.save_dir, opt.name)}/log_sample/{str(epoch + 1).zfill(3)}_{str(iterations).zfill(4)}_{str(opt.name)}.jpg",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

        if (epoch + 1) % opt.save_freq == 0 and opt.local_rank == 0:
            torch.save(
                {
                    "state_dict": warp_model.module.state_dict(),
                    "optim": optimizer.state_dict(),
                },
                opt.save_dir + opt.name + f"/{str(epoch + 1).zfill(3)}_viton_{str(opt.name)}.pt")

        scheduler.step()


if __name__ == '__main__':
    opt = get_opt()
    if opt.local_rank == 0:
        if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
            os.makedirs(os.path.join(opt.save_dir, opt.name))
            os.makedirs(os.path.join(opt.save_dir, opt.name, 'log_sample'))
    train(opt)