import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import motmetrics as mm
from mmtrack.apis import inference_mot, init_model
import cv2
import xml.etree.ElementTree as ET
import torch
import numpy as np
import json
import time
from utils.matching_pure import matching, calculate_cent_corner_pst, draw_matchingpoints, calculate_cent_corner_pst_det
from utils.common import read_xml_r, all_nms, get_matched_ids_frame1, get_matched_ids, A_same_target_refresh_same_ID, \
    B_same_target_refresh_same_ID, same_target_refresh_same_ID, get_matched_trackboxes, get_chosed_track
from utils.trans_matrix import global_compute_transf_matrix as compute_transf_matrix
from utils.supplement import not_matched_supplement, low_confidence_target_refresh_same_ID
import re

parser = ArgumentParser()
# parser.add_argument('--conf+-5ig', default='./configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py',
#                     help='config file')
parser.add_argument('--config', default='./configs/mot/bytetrack/one_carafe_bytetrack_full_mdmt.py',
                    help='config file')

parser.add_argument('--input', default='../Dataset/MDMT/test/1/',
                    help='input video file or folder')

parser.add_argument('--xml_dir', default='../Dataset/MDMT/new_xml/',
                    help='input xml file of the groundtruth')

parser.add_argument('--result_dir', default='./json_resultfiles2/supplement_supplement',
                    help='result_dir name, no "/" in the end')
parser.add_argument('--method', default='NMS-one_carafe_bytetrack_full_mdmt',
                    help='the output directory name used in result_dir')

parser.add_argument(
    '--output', default='./workdirs/', help='output video file (mp4 format) or folder')
parser.add_argument(
    '--output2', default='./workdirs/', help='output video file (mp4 format) or folder')
parser.add_argument('--checkpoint',
                    help='checkpoint file, can be initialized in config files')  # , default="../workdirs/autoassign_epoch_60.pth"
parser.add_argument(
    '--score-thr',
    type=float,
    default=0.0,
    help='The threshold of score to filter bboxes.')
parser.add_argument(
    '--device', default='cuda:0', help='device used for inference')
parser.add_argument(
    '--show',
    # default=True,
    action='store_true',
    help='whether show the results on the fly')
parser.add_argument(
    '--backend',
    choices=['cv2', 'plt'],
    default='cv2',
    help='the backend to visualize the results')
parser.add_argument('--fps', default=10, help='FPS of the output video')
args = parser.parse_args()

model = init_model(args.config, args.checkpoint, device=args.device)
model2 = init_model(args.config, args.checkpoint, device=args.device)

for dirrr in sorted(os.listdir(args.input)):
    if "-2" in dirrr:
        print("dirrr has -2")
        continue
    if "-1" not in dirrr and "-2" not in dirrr:
        continue
    dir2 = dirrr.split('-')[0] + "-2"
    os.makedirs('./matchingimages/' + str(dir2), exist_ok=True)
    # loopp += 1
    # if loopp < 4:
    #     continue
    # print(os.path.join(args.input+dirrr+"/"))
    sequence_dir = os.path.join(args.input + dirrr + "/")
    if osp.isdir(sequence_dir):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                    # os.listdir(args.input)),
                    os.listdir(sequence_dir)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        # imgs = mmcv.VideoReader(args.input)
        imgs = mmcv.VideoReader(sequence_dir)
        IN_VIDEO = True      

    prog_bar = mmcv.ProgressBar(len(imgs))

    matched_ids = []
    A_max_id = 0
    B_max_id = 0

    result_dict = {}
    result_dict2 = {}
    supplement_dict = {}
    supplement_dict2 = {}
    for i, img in enumerate(imgs):
        flag = 0
        coID_confirme = []
        supplement_bbox = np.array([])
        supplement_bbox2 = np.array([])
        if isinstance(img, str):
            # img = osp.join(args.input, img)
            img = osp.join(sequence_dir, img)
            img2 = img.replace("/1/", "/2/")
            img2 = img2.replace("-1", "-2")
            # print(img2)
            image1 = cv2.imread(img)
            image2 = cv2.imread(img2)

        # for the first frame----offline
        if i == 0:
            # print("for the first frame----offline---given labels to update")
            sequence1 = img.split("/")[-2]
            xml_file1 = os.path.join(
                "{}".format(args.xml_dir) + "{}".format(sequence1) + ".xml")
            # print(xml_file1)
            sequence2 = img2.split("/")[-2]
            xml_file2 = os.path.join(
                "{}".format(args.xml_dir) + "{}".format(sequence2) + ".xml")
            # print(xml_file2)
            bboxes1, ids1, labels1 = read_xml_r(xml_file1, i)
            bboxes2, ids2, labels2 = read_xml_r(xml_file2, i)

        max_id = max(A_max_id, B_max_id)
        result, max_id = inference_mot(model, img, frame_id=i, bboxes1=bboxes1, ids1=ids1, labels1=labels1,
                                        max_id=max_id)
        # result = dict(det_bboxes=det_results['bbox_results'],
        #             track_bboxes=track_results['bbox_results'])
        det_bboxes = result['det_bboxes'][0]
        track_bboxes = result['track_bboxes'][0]
        # print("lllllllllllllll",det_bboxes)
        # print("lllllllllllllllffffffffffffffffff", track_bboxes)
        result2, max_id = inference_mot(model2, img2, frame_id=i, bboxes1=bboxes2, ids1=ids2, labels1=labels2,
                                        max_id=max_id)
        det_bboxes2 = result2['det_bboxes'][0]
        track_bboxes2 = result2['track_bboxes'][0]

        
        cent_allclass, corner_allclass = calculate_cent_corner_pst(image1,
                                                                    track_bboxes)  # corner_allclass:ndarray 2n*2        #calculate_cent_corner_pst有更改
        cent_allclass2, corner_allclass2 = calculate_cent_corner_pst(image2,
                                                                        track_bboxes2)  # cent_allclass:ndarray n*2
        if i == 0:
            # 遍历两个track_bboxes，计算匹配点和matchedID
            pts_src, pts_dst, matched_ids = get_matched_ids_frame1(track_bboxes, track_bboxes2, cent_allclass,
                                                                    cent_allclass2)
            A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
            B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))
            # 计算变换矩阵f
            if len(pts_src) >= 5:
                f1, status1 = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
                f2, status2 = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC, 5.0)
                # print(f1, f2)
                f1_last = f1
                f2_last = f2
                # 图像融合可视化
                '''
                leftImage = image1
                rightImage = image2
                resulttt = cv2.warpPerspective(leftImage, f1, (leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
                # 融合方法1
                resulttt.astype(np.float32)
                resulttt = resulttt / 2
                resulttt[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
                cv2.imwrite("./matchingimages/matching{}.jpg".format(i), resulttt)
                '''
                        # 图像融合可视化
                img1 = image1.copy()
                img2 = image2.copy()
                height, width, channels = img1.shape
                img2_transformed = cv2.warpPerspective(img2, f2, (width, height))
                mask = cv2.cvtColor(img2_transformed, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                img1_part = cv2.bitwise_and(img1, img1, mask=mask_inv)

                # 将img1_part和img2_transformed合并
                resultim = cv2.add(img1_part, img2_transformed)
                img2path = osp.join("./matchingimages/{}".format(dir2), f'{i+1:06d}.jpg') 
                print(img2path)
                cv2.imwrite(img2path, resultim)
            # sift 全局匹配（试验）
            # matching(image1, image2, cent_allclass, corner_allclass, 1)

            prog_bar.update()
            continue
        ###########################################################################################################################
        # 第一帧结束，后续帧通过MOT模型进行跟踪，对新产生的ID通过旋转矩阵进行双机匹配（旋转矩阵通过上一帧已匹配目标计算）
        # matched_ids
        # 遍历两个track_bboxes，计算匹配点和matchedID,并得到新增ID及其中心点

        matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
            A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
            = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                corner_allclass2, A_max_id, B_max_id, coID_confirme)
        # print(track_bboxes[:, 0])
        # print(sorted(matched_ids))
        # if len(track_bboxes) != 0:
        #     A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
        #     B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))

        # 计算变换矩阵f
        # print(len(pts_src))
        f1, f1_last = compute_transf_matrix(pts_src, pts_dst, f1_last, image1, image2, dirrr, i)

        # 进行ID更改，将能匹配的新目标赋予同ID(选取旧ID作为关联ID)
        track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = \
            A_same_target_refresh_same_ID(A_new_ID, A_pts, A_pts_corner, f1, cent_allclass2, track_bboxes,
                                            track_bboxes2, matched_ids, det_bboxes, det_bboxes2, image2,
                                            coID_confirme, thres=80)
        if flag == 1:
            print("flag == 1")
            flag = 0
        # 更新result
        result['track_bboxes'][0] = track_bboxes
        result2['track_bboxes'][0] = track_bboxes2
        ####################################################222###################################################
        matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
            A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
            = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                corner_allclass2, A_max_id, B_max_id, coID_confirme)
        f2, f2_last = compute_transf_matrix(pts_dst, pts_src, f2_last, image2, image1, dirrr, i)
        # 图像融合可视化
        img1 = image1.copy()
        img2 = image2.copy()
        height, width, channels = img1.shape
        img2_transformed = cv2.warpPerspective(img2, f2, (width, height))
        mask = cv2.cvtColor(img2_transformed, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_part = cv2.bitwise_and(img1, img1, mask=mask_inv)

        # 将img1_part和img2_transformed合并
        resultim = cv2.add(img1_part, img2_transformed)
        img2path = osp.join("./matchingimages/{}".format(dir2), f'{i+1:06d}.jpg') 
        print(img2path)
        cv2.imwrite(img2path, resultim)
