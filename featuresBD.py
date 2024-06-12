# coding=gbk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import cv2
import random
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import os
from collections import OrderedDict
from torchvision import transforms, models



class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, args, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]

        cap = cv2.VideoCapture(os.path.join(self.videos_dir, video_name))
        background_subtractor = cv2.createBackgroundSubtractorMOG2()

        frames = []
        foreframes = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            foreground_mask = background_subtractor.apply(frame)
            foreframes.append(foreground_mask)

        cap.release()
        video_data = np.stack(frames)
        video_data2 = np.stack(foreframes)

        video_score = self.score[idx]


        transformrgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]

        transformed_video = torch.zeros([video_length, 3, video_height, video_width])
        transformed_gray = torch.zeros([video_length, 3, video_height, video_width])

        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transformrgb(frame)
            transformed_video[frame_idx] = frame

            frame2 = video_data2[frame_idx]
            frame2 = np.dstack((frame2, frame2, frame2))
            frame2 = Image.fromarray(frame2)
            frame2 = transformrgb(frame2)
            transformed_gray[frame_idx] = frame2



        sample = {'video': transformed_video,
                  'videogray': transformed_gray,
                  'score': video_score}

        return sample




class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        model = models.vgg16_bn(pretrained=True)
        self.model_layer = nn.Sequential(*list(model.children())[:-2])
        self.model_layer = nn.Sequential(*list(model.children())[:-2])
        self.model_layer1 = self.model_layer[0][0:7]  # �ӵ�0�㵽��6��
        self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)
        self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)

        self.model_layer2 = self.model_layer[0][7:14]
        self.conv1x1_conv2_spatial = nn.Conv2d(128, 1, 1, bias=True)
        self.conv1x1_conv2_channel_wise = nn.Conv2d(128, 128, 1, bias=True)

        self.model_layer3 = self.model_layer[0][14:24]
        self.conv1x1_conv3_spatial = nn.Conv2d(256, 1, 1, bias=True)
        self.conv1x1_conv3_channel_wise = nn.Conv2d(256, 256, 1, bias=True)

        self.model_layer4 = self.model_layer[0][24:34]

        self.model_layer5 = self.model_layer[0][34:44]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for p in (self.model_layer.parameters()):
            p.requires_grad = False

    def encoder_attention_module_MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)

        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def pooling_and_std(self, x):
        xp1 = nn.functional.adaptive_avg_pool2d(x, 1)
        xs1 = global_std_pool2d(x)
        xps1 = torch.cat((xp1, xs1), 1)
        return xps1

    def forward(self, x_rgb, x_fd):
        #x_fd·
        y1 = self.model_layer1(x_fd)  # torch.Size([2, 64, 120, 120])
        y2 = self.model_layer2(y1)  # torch.Size([2, 128, 60, 60])

        y3 = self.model_layer3(y2)  # torch.Size([2, 256, 30, 30])
        y3_pool = self.pooling_and_std(y3)
        # print("xy", xy.shape)  #torch.Size([16, 256, 67, 120])
        y4 = self.model_layer4(y3)  # torch.Size([2, 512, 15, 15])
        y4_pool = self.pooling_and_std(y4)
        y5 = self.model_layer5(y4)  # torch.Size([2, 512, 7, 7])
        y5_pool = self.pooling_and_std(y5)
        yfeature_all = torch.cat((y3_pool, y4_pool, y5_pool), 1)

        # x_rgb·��
        x1 = self.model_layer1(x_rgb)  # torch.Size([2, 64, 120, 120])
        x1_1 = self.encoder_attention_module_MGA_tmc(x1, y1, self.conv1x1_conv1_channel_wise, self.conv1x1_conv1_spatial)
        x2 = self.model_layer2(x1_1)
        x2_1 = self.encoder_attention_module_MGA_tmc(x2, y2, self.conv1x1_conv2_channel_wise,self.conv1x1_conv2_spatial)

        x3 = self.model_layer3(x2_1)
        x3_pool = self.pooling_and_std(x3)
        x4 = self.model_layer4(x3)  # torch.Size([2, 512, 15, 15])
        x4_pool = self.pooling_and_std(x4)
        x5 = self.model_layer5(x4)  # torch.Size([2, 512, 7, 7])
        x5_pool = self.pooling_and_std(x5)
        xfeature_all = torch.cat((x3_pool, x4_pool, x5_pool), 1)

        return xfeature_all, yfeature_all

def global_std_pool2d(x):
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)

import logging
def get_features(video_data, current_videodf, frame_batch_size=16, device='cuda'):
    """feature extraction"""

    extractor1 = VGG16().to(device)

    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output11 = torch.Tensor().to(device)
    output31 = torch.Tensor().to(device)

    extractor1.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch1 = video_data[frame_start:frame_end].to(device)
            batch3 = current_videodf[frame_start:frame_end].to(device)
            features1, features2 = extractor1(batch1, batch3)
            output11 = torch.cat((output11, features1), 0)  
            output31 = torch.cat((output31, features2), 0)  

            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch1 = video_data[frame_start:video_length].to(device)
        last_batch3 = current_videodf[frame_start:video_length].to(device)
        features1, features2 = extractor1(last_batch1, last_batch3)
        output11 = torch.cat((output11, features1), 0)
        output31 = torch.cat((output31, features2), 0)

        print("output11:", output11.shape)  # 736
    return output11, output31

if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='KoNViD-1k', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=16,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--width', default=336, type=int, help='width of RGB image')
    parser.add_argument('--height', default=336, type=int, help='height of RGB image')
    parser.add_argument('--model', default='Models.SAMNet', type=str, help='which model to test')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')

    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    """
    Data1_train01:
    """

    if args.database == 'KoNViD-1k':
        videos_dir = '../dataset/KoNViD_1k/'  # videos dir
        features_dir ='CNN_features_KoNViD-1k/'   # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = '../dataset/CVD2014/'
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE_VQC':
        videos_dir = '../dataset/LIVE_VQC/'
        features_dir = 'CNN_features_LIVE_VQC/'
        datainfo = 'data/LIVE-VQCinfo.mat'


    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print("cuda is available", torch.cuda.is_available())

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(args, videos_dir, video_names, scores, video_format, width, height)
    #a = int(len(dataset) / 2)
    for i in range(len(dataset)):
        if os.path.exists(features_dir + str(i) + '_score'):
            print("exists feature{},skip".format(i))
            continue
        current_data = dataset[i]
        current_video = current_data['video']
        #current_videolc = current_data['videolc']  #torch.Size([240, 4, 540, 960])
        current_videodf = current_data['videogray']
        current_score = current_data['score']
        #print("curren--:", current_score.shape)
        print('Video {}: length {}'.format(i, current_video.shape[0]))

        output1, output2 = get_features(current_video,current_videodf, args.frame_batch_size, device)

        np.save(features_dir + str(i) + '_VGG19_rgb', output1.to('cpu').numpy())
        np.save(features_dir + str(i) + '_VGG19_gray', output2.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)

