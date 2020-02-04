import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

from dataset.dataloader import ValidationLoader
from network.DINet import get_model


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluation Process")
    parser.add_argument("--root", type=str, default='')
    parser.add_argument("--data-list", type=str, default='./dataset/list/val_id.txt')
    parser.add_argument("--crop-size", type=int, default=513)
    parser.add_argument("--node1-cls", type=int, default=20)
    parser.add_argument("--node2-cls", type=int, default=3)
    parser.add_argument("--node3-cls", type=int, default=2)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--restore-from", type=str, default='')
    parser.add_argument("--is-mirror", action="store_true")
    parser.add_argument('--eval-scale', nargs='+', type=float, default=[0.5, 0.75, 1.0, 1.25, 1.50])
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # initialization
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    # conduct model & load pre-trained weights
    model = get_model(num_classes=args.num_classes)
    restore_from = args.restore_from
    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval().cuda()
    # data loader
    testloader = data.DataLoader(ValidationLoader(args.root, args.data_list, crop_size=args.crop_size),
                                 batch_size=1, shuffle=False, pin_memory=True)

    matrix_n1 = np.zeros((args.node1_cls, args.node1_cls))
    matrix_n2 = np.zeros((args.node2_cls, args.node2_cls))
    matrix_n3 = np.zeros((args.node3_cls, args.node3_cls))

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d images have been proceeded' % index)
        image, label_n1, label_n2, label_n3, ori_size, name = batch

        ori_size = ori_size[0].numpy()
        output_n1, output_n2, output_n3 = predict(model, image.numpy(),
                                                  (np.asscalar(ori_size[0]), np.asscalar(ori_size[1])),
                                                  is_mirror=args.is_mirror, scales=args.eval_scale)

        pred_n1 = np.asarray(np.argmax(output_n1, axis=2), dtype=np.uint8)
        pred_n2 = np.asarray(np.argmax(output_n2, axis=2), dtype=np.uint8)
        pred_n3 = np.asarray(np.argmax(output_n3, axis=2), dtype=np.uint8)

        gt_n1 = np.asarray(label_n1[0].numpy(), dtype=np.int)
        gt_n2 = np.asarray(label_n2[0].numpy(), dtype=np.int)
        gt_n3 = np.asarray(label_n3[0].numpy(), dtype=np.int)

        ignore_index = gt_n1 != 255
        gt_n1 = gt_n1[ignore_index]
        pred_n1 = pred_n1[ignore_index]
        gt_n2 = gt_n2[ignore_index]
        pred_n2 = pred_n2[ignore_index]
        gt_n3 = gt_n3[ignore_index]
        pred_n3 = pred_n3[ignore_index]

        matrix_n1 += get_confusion_matrix(gt_n1, pred_n1, args.node1_cls)
        matrix_n2 += get_confusion_matrix(gt_n2, pred_n2, args.node2_cls)
        matrix_n3 += get_confusion_matrix(gt_n3, pred_n3, args.node3_cls)

    # node1 segmentation
    pos = matrix_n1.sum(1)
    res = matrix_n1.sum(0)
    tp = np.diag(matrix_n1)
    pixel_accuracy = tp.sum() / pos.sum()
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    print('N1 --> Pixel accuracy: %f \n' % pixel_accuracy)
    print('N1 --> Mean accuracy: %f \n' % mean_accuracy)
    print('N1 --> Mean IU: %f \n' % mean_IU)
    for index, IU in enumerate(IU_array):
        print('%f ', IU)
    print('------------------------------------')
    # node2 segmentation
    pos_n2 = matrix_n2.sum(1)
    res_n2 = matrix_n2.sum(0)
    tp_n2 = np.diag(matrix_n2)
    pixel_accuracy_n2 = tp_n2.sum() / pos_n2.sum()
    mean_accuracy_n2 = (tp_n2 / np.maximum(1.0, pos_n2)).mean()
    IU_array_n2 = (tp_n2 / np.maximum(1.0, pos_n2 + res_n2 - tp_n2))
    mean_IU_n2 = IU_array_n2.mean()
    print('N2 --> Pixel accuracy: %f \n' % pixel_accuracy_n2)
    print('N2 --> Mean accuracy: %f \n' % mean_accuracy_n2)
    print('N2 --> Mean IU: %f \n' % mean_IU_n2)
    for index, IU in enumerate(IU_array_n2):
        print('%f ', IU)
    print('------------------------------------')
    # node3 segmentation
    pos_n3 = matrix_n3.sum(1)
    res_n3 = matrix_n3.sum(0)
    tp_n3 = np.diag(matrix_n3)
    pixel_accuracy_n3 = tp_n3.sum() / pos_n3.sum()
    mean_accuracy_n3 = (tp_n3 / np.maximum(1.0, pos_n3)).mean()
    IU_array_n3 = (tp_n3 / np.maximum(1.0, pos_n3 + res_n3 - tp_n3))
    mean_IU_n3 = IU_array_n3.mean()
    print('N3 --> Pixel accuracy: %f \n' % pixel_accuracy_n3)
    print('N3 --> Mean accuracy: %f \n' % mean_accuracy_n3)
    print('N3 --> Mean IU: %f \n' % mean_IU_n3)
    for index, IU in enumerate(IU_array_n3):
        print('%f ', IU)
    print('------------------------------------')


def predict(net, image, output_size, is_mirror=True, scales=[1]):
    if is_mirror:
        image_rev = image[:, :, :, ::-1]

    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)

    outputs, outputs_hb, outputs_fb = [], [], []
    if is_mirror:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image[0, :, :, :]
                image_rev_scale = image_rev[0, :, :, :]

            image_scale = np.stack((image_scale, image_rev_scale))

            with torch.no_grad():
                prediction = net(Variable(torch.from_numpy(image_scale)).cuda())
                part_pred = interp(prediction[0]).cpu().data.numpy()
                hb_pred = interp(prediction[1]).cpu().data.numpy()
                fb_pred = interp(prediction[2]).cpu().data.numpy()
            # node 1
            part_rev = part_pred[1, :, :, :].copy()
            part_rev = part_rev[:, :, ::-1]
            part_norm = part_pred[0, :, :, :]
            part_prediction = np.mean([part_norm, part_rev], axis=0)
            outputs.append(part_prediction)
            # node 2
            hb_rev = hb_pred[1, :, :, :].copy()
            hb_rev = hb_rev[:, :, ::-1]
            hb_norm = hb_pred[0, :, :, :]
            hb_prediction = np.mean([hb_norm, hb_rev], axis=0)
            outputs_hb.append(hb_prediction)
            # node 3
            fb_rev = fb_pred[1, :, :, :].copy()
            fb_rev = fb_rev[:, :, ::-1]
            fb_norm = fb_pred[0, :, :, :]
            fb_prediction = np.mean([fb_norm, fb_rev], axis=0)
            outputs_fb.append(fb_prediction)

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1, 2, 0)
        outputs_hb = np.mean(outputs_hb, axis=0)
        outputs_hb = outputs_hb.transpose(1, 2, 0)
        outputs_fb = np.mean(outputs_fb, axis=0)
        outputs_fb = outputs_fb.transpose(1, 2, 0)

    else:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
            else:
                image_scale = image[0, :, :, :]

            with torch.no_grad():
                prediction = net(Variable(torch.from_numpy(image_scale).unsqueeze(0)).cuda())
                part_pred = interp(prediction[0]).cpu().data.numpy()
                hb_pred = interp(prediction[1]).cpu().data.numpy()
                fb_pred = interp(prediction[2]).cpu().data.numpy()
                prediction = part_pred
                hb_prediction = hb_pred
                fb_prediction = fb_pred

            outputs.append(prediction[0, :, :, :])
            outputs_hb.append(hb_prediction[0, :, :, :])
            outputs_fb.append(fb_prediction[0, :, :, :])

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1, 2, 0)
        outputs_hb = np.mean(outputs_hb, axis=0)
        outputs_hb = outputs_hb.transpose(1, 2, 0)
        outputs_fb = np.mean(outputs_fb, axis=0)
        outputs_fb = outputs_fb.transpose(1, 2, 0)

    return outputs, outputs_hb, outputs_fb


def scale_image(image, scale):
    image = image[0, :, :, :]
    image = image.transpose((1, 2, 0))
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = image.transpose((2, 0, 1))
    return image


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calculate the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the number of category
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


if __name__ == '__main__':
    main()
