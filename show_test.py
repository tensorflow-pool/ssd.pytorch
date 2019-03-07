from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.autograd import Variable

from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import VOC_ROOT, VOC_CLASSES as labelmap
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/3_7_VOC_440.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'test1.txt'
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, mode='w') as f:
        num_images = len(testset)
        for i in range(num_images):
            print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
            img = testset.pull_image(i)
            img_id, annotation = testset.pull_anno(i)
            x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))

            f.write('\nGROUND TRUTH FOR: ' + img_id + '\n')
            for box in annotation:
                f.write('label gt: ' + ' || '.join(str(b) for b in box) + labelmap[int(box[-1])] + '\n')

            if cuda:
                x = x.cuda()

            y = net(x)  # forward pass
            detections = y.data
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])
            pred_num = 0
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.4:
                    if pred_num == 0:
                        f.write('PREDICTIONS: ' + '\n')
                    score = detections[0, i, j, 0]
                    label_name = labelmap[i - 1]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    pred_num += 1
                    f.write(str(pred_num) + ' label: ' + label_name + ' score: ' + str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                    j += 1

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
            plt.title("")  # 第一幅图片标题
            plt.imshow(img)  # 绘制第一幅图片
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            currentAxis = plt.gca()
            for i, rects in enumerate(detections[0]):
                h, w, _ = img.shape

                for rect in rects:
                    if rect[0] < 0.2:
                        continue
                    pt = (rect[1] * w, rect[2] * h)
                    coords = pt, (rect[3] - rect[1]) * w, (rect[4] - rect[2]) * h
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=colors[i], linewidth=2))
                    display_txt = '%s_%s' % (labelmap[i - 1], str(rect[0]))
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': colors[i], 'alpha': 0.5})

            plt.show()


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'trainval')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()
