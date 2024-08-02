import os
import random
import torchvision.transforms as transforms
from .base_dataset import BaseDataset
from PIL import Image, ImageDraw
import torch
import numpy as np
import cv2
import pycocotools.mask as maskUtils
import math
import json
from torchvision import utils as vutils


def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)


class AlignedDataset(BaseDataset):
    def __init__(self, dataroot, resolution=512, mode='train', unpaired=False):
        super(AlignedDataset, self).__init__()
        self.root = dataroot
        self.mode = mode
        self.unpaired = unpaired
        self.toTensor = transforms.ToTensor()
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

        if resolution == 512:
            self.load_height = 512
            self.load_width = 384
            self.radius = 8
        else:
            self.load_height = 1024
            self.load_width = 768
            self.radius = 16
        self.crop_size = (self.load_height, self.load_width)

        if mode == 'train':
            pair_txt_path = os.path.join(self.root, 'train_pairs_1018.txt')
        elif self.unpaired:
            pair_txt_path = os.path.join(self.root, 'test_pairs_unpaired_1018.txt')
        else:
            pair_txt_path = os.path.join(self.root, 'test_pairs_paired_1018.txt')
        with open(pair_txt_path, 'r') as f:
            lines = f.readlines()

        im_names = []
        c_names = []
        for line in lines:
            im_name, c_name, c_type = line.strip().split()
            im_names.append(im_name)
            c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

        self.dataset_size = len(self.im_names)

    ############### get palm mask ################
    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b - d) / 4, b + (c - a) / 4
        x2, y2 = a - (b - d) / 4, b - (c - a) / 4

        x3, y3 = c + (b - d) / 4, d + (c - a) / 4
        x4, y4 = c - (b - d) / 4, d - (c - a) / 4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c - a, d - b
        v1_x, v1_y = x3 - x1, y3 - y1
        v2_x, v2_y = x4 - x1, y4 - y1

        cos1 = (v0_x * v1_x + v0_y * v1_y) / \
               (math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v1_x * v1_x + v1_y * v1_y))
        cos2 = (v0_x * v2_x + v0_y * v2_y) / \
               (math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v2_x * v2_x + v2_y * v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        # shoulder, elbow, wrist
        s_x, s_y, s_c = hand_keypoints[0]
        e_x, e_y, e_c = hand_keypoints[1]
        w_x, w_y, w_c = hand_keypoints[2]

        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            if self.load_height == 512:
                kernel = np.ones((50, 50), np.uint8)
            else:
                kernel = np.ones((100, 100), np.uint8)
            up_mask = cv2.dilate(up_mask, kernel, iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            if self.load_height == 512:
                kernel = np.ones((30, 30), np.uint8)
            else:
                kernel = np.ones((60, 60), np.uint8)
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask + hand_bottom_mask)
                             == 2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, parsing, keypoints):
        h, w = parsing.shape[0:2]

        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()

        left_hand_up_mask, left_hand_bottom_mask = self.get_hand_mask(
            left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_bottom_mask = self.get_hand_mask(
            right_hand_keypoints, h, w)

        # mask refined by parsing
        left_hand_mask = (parsing == 15).astype(np.float32)
        right_hand_mask = (parsing == 16).astype(np.float32)

        left_palm_mask = self.get_palm_mask(
            left_hand_mask, left_hand_up_mask, left_hand_bottom_mask)
        right_palm_mask = self.get_palm_mask(
            right_hand_mask, right_hand_up_mask, right_hand_bottom_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    def __getitem__(self, index):
        # C_type = self.C_types[index]
        if self.unpaired:
            key = 'unpaired'
        else:
            key = 'paired'

        # person image
        P_path = os.path.join(self.root, self.mode, 'image', self.im_names[index])
        # P = transforms.Resize(self.crop_size, interpolation=2)(Image.open(P_path).convert('RGB'))
        P = Image.open(P_path).convert('RGB')
        P_tensor = self.rgb_transform(P)

        # person 2d pose
        pose_path = P_path.replace('image', 'openpose_json')[:-4] + '_keypoints.json'
        with open(pose_path, 'r') as f:
            datas = json.load(f)
        pose_data = np.array(datas['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

        # person parsing
        parsing_path = P_path.replace('image', 'parse-bytedance')[:-4] + '.png'
        parsing = Image.open(parsing_path).convert('L')
        parsing_tensor = self.toTensor(parsing) * 255.0

        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8)
        palm_mask_np = self.get_palm(parsing_np, pose_data)
        palm_mask = torch.tensor(palm_mask_np.transpose(2, 0, 1)).float()

        # clothes
        C_path = os.path.join(self.root, self.mode, 'cloth', self.c_names[key][index])
        # C = transforms.Resize(self.crop_size, interpolation=2)(Image.open(C_path).convert('RGB'))
        C = Image.open(C_path).convert('RGB')
        C_tensor = self.rgb_transform(C)

        CM_path = C_path.replace('cloth', 'cloth_mask-bytedance')[:-4] + '.png'
        # CM = transforms.Resize(self.crop_size, interpolation=0)(Image.open(CM_path).convert('L'))
        CM = Image.open(CM_path).convert('L')
        CM_tensor = self.toTensor(CM)

        mask = np.array([(parsing_np == index).astype(int) for index in [5, 6, 11, 15, 16, 21, 22, 24, 25]])
        mask = np.sum(mask, axis=0)
        kernel_size = int(5 * (self.load_width / 256))
        mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=3)
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=1)
        mask = mask.astype(np.float32)
        inpaint_mask = 1 - self.toTensor(mask)

        W_path = P_path.replace('image', 'cloth-warp' if not self.unpaired else 'unpaired-cloth-warp')
        # W_tensor = transforms.Resize(self.crop_size, interpolation=2)(Image.open(W_path))
        W_tensor = Image.open(W_path)
        W_tensor = self.rgb_transform(W_tensor)
        feat = W_tensor * (1 - inpaint_mask) + P_tensor * inpaint_mask

        down, up, left, right = mask2bbox(CM_tensor[0].numpy())
        ref_image = C_tensor[:, down:up, left:right]
        ref_image = (ref_image + 1.0) / 2.0
        ref_image = transforms.Resize((224, 224))(ref_image)
        ref_image = self.clip_normalize(ref_image)

        inpaint = feat * (1 - palm_mask) + P_tensor * palm_mask

        input_dict = {
            'GT': P_tensor,
            "inpaint_image": inpaint,
            "inpaint_mask": inpaint_mask,
            'warp_feat': feat,
            'ref_imgs': ref_image,
            'file_name': P_path.split('/')[-1],
        }
        # vutils.save_image(input_dict['GT'], P_path.replace('test/image', 'pair_gt_500'), normalize=True)

        return input_dict

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'AlignedDataset'