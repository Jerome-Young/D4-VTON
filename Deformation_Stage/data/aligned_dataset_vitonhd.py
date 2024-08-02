import os
from random import random
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image, ImageDraw
import torch
import numpy as np
from torchvision import transforms
import cv2
import pycocotools.mask as maskUtils
import math
import json


class AlignedDataset(BaseDataset):
    def initialize(self, opt, mode='train'):
        self.opt = opt
        self.root = opt.dataroot
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        if self.load_height == 512:
            self.radius = 8
        else:
            self.radius = 16

        pair_txt_path = os.path.join(self.root, opt.image_pairs_txt)
        if mode == 'train' and 'train' in opt.image_pairs_txt:
            self.mode = 'train'
        else:
            self.mode = 'test'
        with open(pair_txt_path, 'r') as f:
            lines = f.readlines()

        self.P_paths = []
        self.C_paths = []
        self.C_types = []
        for line in lines:
            p_name, c_name, c_type = line.strip().split()
            P_path = os.path.join(self.root, self.mode, 'image', p_name)
            C_path = os.path.join(self.root, self.mode, 'cloth', c_name)
            if self.load_height == 1024:
                P_path = P_path.replace('.png', '.jpg')
                C_path = C_path.replace('.png', '.jpg')
            self.P_paths.append(P_path)
            self.C_paths.append(C_path)
            self.C_types.append(c_type)

        self.dataset_size = len(self.P_paths)

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

    def get_img_agnostic(self, img, parse, pose_data, parse_palm):
        pose_data = pose_data[:, :2]
        parse_array = np.array(parse)
        parse_head = ((parse_array == 1).astype(np.float32) +
                      (parse_array == 2).astype(np.float32) +
                      (parse_array == 4).astype(np.float32) +
                      (parse_array == 14).astype(np.float32))
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 21).astype(np.float32) +
                       (parse_array == 22).astype(np.float32))
        parse_lower = ((parse_array == 8).astype(np.float32) +
                       (parse_array == 9).astype(np.float32) +
                       (parse_array == 10).astype(np.float32) +
                       (parse_array == 13).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32) +
                       (parse_array == 20).astype(np.float32) +
                       (parse_array == 23).astype(np.float32) +
                       (parse_array == 26).astype(np.float32) +
                       (parse_array == 27).astype(np.float32) +
                       (parse_array == 28).astype(np.float32))

        parse_palm = parse_palm.astype(np.float32).squeeze()
        r = 10
        img = np.array(img)
        img[parse_upper > 0, :] = 0
        img = Image.fromarray(img)
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2] + 1e-8)
        length_b = np.linalg.norm(pose_data[12] - pose_data[9] + 1e-8)
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'black', width=r * 10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3), 'black', 'black')

        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
            pointx, pointy = pose_data[i]
            if i in [4, 7]:
                pass
            else:
                agnostic_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black',
                                      'black')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'black', 'black')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'black', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'black', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'black', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'black', 'black')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 7, pointy - r * 7, pointx + r * 7, pointy + r * 7), 'black', 'black')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_palm * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        C_type = self.C_types[index]

        # person image
        P_path = self.P_paths[index]
        P = Image.open(P_path).convert('RGB')
        P_np = np.array(P)
        params = get_params(self.opt, P.size)
        transform_for_rgb = get_transform(self.opt, params)
        P_tensor = transform_for_rgb(P)

        # person 2d pose
        pose_path = P_path.replace('/image/', '/openpose_json/')[:-4] + '_keypoints.json'
        with open(pose_path, 'r') as f:
            datas = json.load(f)
        pose_data = np.array(datas['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.load_height, self.load_width)
        r = self.radius
        im_pose = Image.new('L', (self.load_width, self.load_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.load_width, self.load_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = transform_for_rgb(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        Pose_tensor = pose_map

        # person 3d pose
        densepose_path = P_path.replace('/image/', '/densepose/')[:-4] + '.png'
        dense_mask = Image.open(densepose_path).convert('L')
        transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        dense_mask_tensor = transform_for_mask(dense_mask) * 255.0
        dense_mask_tensor = dense_mask_tensor[0:1, ...]

        # person parsing
        parsing_path = P_path.replace('/image/', '/parse-bytedance/')[:-4] + '.png'
        parsing = Image.open(parsing_path).convert('L')
        parsing_tensor = transform_for_mask(parsing) * 255.0

        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8)
        palm_mask_np = self.get_palm(parsing_np, pose_data)

        person_clothes_left_sleeve_mask_np = (parsing_np == 21).astype(int) + \
                                             (parsing_np == 24).astype(int)
        person_clothes_torso_mask_np = (parsing_np == 5).astype(int) + \
                                       (parsing_np == 6).astype(int)
        person_clothes_right_sleeve_mask_np = (parsing_np == 22).astype(int) + \
                                              (parsing_np == 25).astype(int)
        person_clothes_mask_np = person_clothes_left_sleeve_mask_np + \
                                 person_clothes_torso_mask_np + \
                                 person_clothes_right_sleeve_mask_np
        hand_mask_np = (parsing_np==15).astype(int) + (parsing_np==16).astype(int)
        neck_mask_np = (parsing_np==11).astype(int)

        person_clothes_mask_tensor = torch.tensor(person_clothes_mask_np.transpose(2, 0, 1)).float()
        # person_clothes_left_sleeve_mask_tensor = torch.tensor(
        #     person_clothes_left_sleeve_mask_np.transpose(2, 0, 1)).float()
        # person_clothes_torso_mask_tensor = torch.tensor(person_clothes_torso_mask_np.transpose(2, 0, 1)).float()
        # person_clothes_right_sleeve_mask_tensor = torch.tensor(
        #     person_clothes_right_sleeve_mask_np.transpose(2, 0, 1)).float()
        #
        # background_mask_tensor = 1 - (person_clothes_left_sleeve_mask_tensor + person_clothes_torso_mask_tensor +
        #                               person_clothes_right_sleeve_mask_tensor)

        # preserve region mask
        preserve_mask_np = np.array([(parsing_np == index).astype(int) for index in
                                     [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27, 28]])
        preserve_mask_np = np.sum(preserve_mask_np, axis=0)
        preserve_mask_np = preserve_mask_np + palm_mask_np


        preserve_mask_tensor = torch.tensor(preserve_mask_np.transpose(2, 0, 1)).float()

        # clothes
        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_for_rgb(C)

        CM_path = C_path.replace('/cloth/', '/cloth_mask-bytedance/')[:-4] + '.png'
        CM = Image.open(CM_path).convert('L')
        CM_tensor = transform_for_mask(CM)

        input_dict = {
            'image': P_tensor, 'pose': Pose_tensor, 'densepose': dense_mask_tensor,
            'person_clothes_mask': person_clothes_mask_tensor,
            # 'person_clothes_left_mask': person_clothes_left_sleeve_mask_tensor,
            # 'person_clothes_middle_mask': person_clothes_torso_mask_tensor,
            # 'person_clothes_right_mask': person_clothes_right_sleeve_mask_tensor,
            # 'background_mask': background_mask_tensor,
            'preserve_mask': preserve_mask_tensor,
            'color': C_tensor, 'edge': CM_tensor,
            'c_type': C_type,
            'color_path': C_path,
            'img_path': P_path,
        }

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return len(self.P_paths) // (self.opt.batch_size * self.opt.num_gpus) * (
                        self.opt.batch_size * self.opt.num_gpus)
        else:
            return len(self.P_paths)

    def name(self):
        return 'AlignedDataset'