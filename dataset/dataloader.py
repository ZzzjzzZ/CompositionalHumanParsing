import os
import os.path

import cv2
import numpy as np
import torch.utils.data as data


class ValidationLoader(data.Dataset):
    def __init__(self, root, list_path, crop_size):

        fid = open(list_path, 'r')
        imgs, segs = [], []
        for line in fid.readlines():
            idx = line.strip().split(' ')[0]
            image_path = os.path.join(root, 'images/' + str(idx) + '.jpg')
            seg_path = os.path.join(root, 'segmentations/' + str(idx) + '.png')
            imgs.append(image_path)
            segs.append(seg_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size

    def __getitem__(self, index):
        # load data
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)
        ori_size = img.shape

        h, w = seg.shape
        length = max(w, h)
        ratio = self.crop_size / length
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype(np.float32) - mean
        img = img.transpose((2, 0, 1))

        images = img.copy()
        label_n1 = seg.copy()
        label_n2 = seg.copy()
        label_n2[(label_n2 > 0) & (label_n2 <= 7)] = 1
        label_n2[(label_n2 > 7) & (label_n2 <= 10)] = 2
        label_n2[label_n2 == 11] = 1
        label_n2[label_n2 == 12] = 2
        label_n2[(label_n2 > 12) & (label_n2 <= 15)] = 1
        label_n2[(label_n2 > 15) & (label_n2 < 255)] = 2
        label_n3 = seg.copy()
        label_n3[(label_n3 > 0) & (label_n3 < 255)] = 1

        return images, label_n1, label_n2, label_n3, np.array(ori_size), name

    def __len__(self):
        return len(self.imgs)
