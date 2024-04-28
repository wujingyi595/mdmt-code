import numpy as np

import torch

import kornia
from kornia.feature.laf import (
    laf_from_center_scale_ori, extract_patches_from_pyramid)

import pycolmap

EPS = 1e-6


def sift_to_rootsift(x):
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


class Dog:
    def __init__(self, nfeatures=-1, descriptor='sift', patch_size=32, mr_size=12):
        self.nfeatures = nfeatures
        self.descriptor = descriptor
        self.patch_size = patch_size
        self.mr_size = mr_size

        if descriptor == 'sosnet':
            self.describe = kornia.feature.SOSNet(pretrained=True)
            if torch.cuda.is_available():
                self.describe.cuda()
        elif descriptor == 'hardnet':
            self.describe = kornia.feature.HardNet(pretrained=True)
            if torch.cuda.is_available():
                self.describe.cuda()
        elif descriptor not in ['sift', 'rootsift']:
            raise ValueError(f'Unknown descriptor: {descriptor}')

        use_gpu = pycolmap.has_cuda
        options = {
            'edge_threshold':10,
            'octave_resolution':4, 
            'first_octave': 0,
            'peak_threshold': 0.01,  #对比度阈值原始为0.01
        }
        if descriptor == 'rootsift':
            options['normalization'] = pycolmap.Normalization.L1_ROOT
        else:
            options['normalization'] = pycolmap.Normalization.L2
        self.sift = pycolmap.Sift(
            options=pycolmap.SiftExtractionOptions(options),
            device=getattr(pycolmap.Device, 'cuda' if use_gpu else 'cpu'))

    def detectAndCompute(self, img):
        keypoints, scores, descriptors = self.sift.extract(img)
        nan_count = np.isnan(descriptors).sum()
        # if(nan_count>0):
        #     print("矩阵中NaN值的个数：", nan_count) 
        #     descriptors[np.isnan(descriptors)] = 0.0
        # print("111111", descriptors, keypoints.shape[0])
        
        scales = keypoints[:, 2]
        oris = np.rad2deg(keypoints[:, 3])


        if self.descriptor in ['sift', 'rootsift']:
            # We still renormalize because COLMAP does not normalize well,
            # maybe due to numerical errors
            # print("111115", descriptors)
            if self.descriptor == 'rootsift':
                # print("22222", descriptors)
                descriptors = sift_to_rootsift(descriptors)
                print('print sift to rootsift')
        elif self.descriptor in ('sosnet', 'hardnet'):
            image = torch.from_numpy(img)
            image = image.view(1, 1, image.shape[0], image.shape[1])
            if torch.cuda.is_available():
                image = image.cuda()
            center = keypoints[:, :2] + 0.5
            laf_scale = scales * self.mr_size / 2
            laf_ori = -oris
            lafs = laf_from_center_scale_ori(
                torch.from_numpy(center)[None],
                torch.from_numpy(laf_scale)[None, :, None, None],
                torch.from_numpy(laf_ori)[None, :, None]).to(image.device)
            patches = extract_patches_from_pyramid(
                image, lafs, PS=self.patch_size)[0]
            descriptors = patches.new_zeros((len(patches), 128))
            if len(patches) > 0:
                for start_idx in range(0, len(patches), 1024):
                    end_idx = min(len(patches), start_idx + 1024)
                    descriptors[start_idx:end_idx] = self.describe(
                        patches[start_idx:end_idx])
            descriptors = descriptors.cpu().detach().numpy()
        else:
            raise ValueError(f'Unknown descriptor: {self.descriptor}')

        if self.nfeatures != -1 and keypoints.shape[0] > self.nfeatures:
            # print("5555555", descriptors)
            indices = scores.argsort()[::-1]
            keypoints, descriptors, scores = keypoints[indices], descriptors [indices], scores[indices]
            keypoints, descriptors, scores = keypoints[:self.nfeatures], descriptors [:self.nfeatures], scores[:self.nfeatures]

        #     print("66666", descriptors)
        # print("9999999", descriptors)
        return keypoints, scores, descriptors
