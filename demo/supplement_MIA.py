# _*_ coding: utf-8 _*_
# @Time    :2022/7/12 16:38
# @Author  :LiuZhihao
# @File    :supplement.py

# Copyright (c) OpenMMLab. All rights reserved.
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
from utils.trans_matrix import supp_compute_transf_matrix as compute_transf_matrix
from utils.supplement import not_matched_supplement, low_confidence_target_refresh_same_ID
import re


###############################D-S


# 初始化基本信任分配 用于D-S
def initialize_mass_function(confidence):
    return {

        'target': confidence,
        'not_target': 1 - confidence,
        'uncertain': 0.0  # 如果有不确定性，可以适当调整
    }


# 融合两个信任分配
def combine_mass_functions(m1, m2):
    keys = m1.keys()
    combined_m = {}
    conflict = 0
    for key in keys:
        combined_m[key] = sum(m1[k1] * m2[k2] for k1 in keys for k2 in keys if k1 == k2 and k1 == key)
    total_mass = sum(combined_m.values())
    if total_mass < 1:  # 如果存在冲突
        conflict = 1 - total_mass
    for key in keys:
        if total_mass > 0:
            combined_m[key] /= total_mass  # 重新归一化以考虑冲突
    return combined_m, conflict


def main():
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
    assert args.output or args.show
    loopp = 0
    # load images
    track_bboxes_old = []
    track_bboxes2_old = []
    time_start_all = time.time()
    for dirrr in sorted(os.listdir(args.input)):
        if "-2" in dirrr:
            print("dirrr has -2")
            continue
        if "-1" not in dirrr and "-2" not in dirrr:
            continue
        dir2 = dirrr.split('-')[0] + "-2"
        os.makedirs('./matchingimages/' + str(dirrr), exist_ok=True)
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
        # define output
        if args.output is not None:
            # if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            # out_dir = tempfile.TemporaryDirectory()
            out_path = './workdirs/'
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
            # else:
            #     OUT_VIDEO = False
            #     out_path = args.output
            #     os.makedirs(out_path, exist_ok=True)
            outputname = args.output + dirrr + 'A.mp4'
        if args.output2 is not None:
            # if args.output2.endswith('.mp4'):
            OUT_VIDEO = True
            # out_dir2 = tempfile.TemporaryDirectory()
            # out_path2 = out_dir2.name
            _out2 = args.output2.rsplit(os.sep, 1)
            if len(_out2) > 1:
                os.makedirs(_out2[0], exist_ok=True)
            # else:
            #     OUT_VIDEO = False
            #     out_path2 = args.output2
            #     os.makedirs(out_path2, exist_ok=True)
            output2name = args.output + dirrr + 'B.mp4'

        fps = args.fps
        if args.show or OUT_VIDEO:
            if fps is None and IN_VIDEO:
                fps = imgs.fps
            if not fps:
                raise ValueError('Please set the FPS for the output video.')
            fps = int(fps)

        # build the model from a config file and a checkpoint file
        # print(args.checkpoint)
        model = init_model(args.config, args.checkpoint, device=args.device)
        model2 = init_model(args.config, args.checkpoint, device=args.device)

        prog_bar = mmcv.ProgressBar(len(imgs))

        matched_ids = []
        A_max_id = 0
        B_max_id = 0

        result_dict = {}
        result_dict2 = {}
        supplement_dict = {}
        supplement_dict2 = {}
        time_start = time.time()
        # test and show/save the images
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
                # print(bboxes1)

                # 第一帧做完之后不进行后续操作，后续操作从第二帧开始
                # continue
            # print("4444444444444333333333333333333")
            # inference process
            max_id = max(A_max_id, B_max_id)
            result, max_id = inference_mot(model, img, frame_id=i, bboxes1=bboxes1, ids1=ids1, labels1=labels1,
                                           max_id=max_id)
            # result = dict(det_bboxes=det_results['bbox_results'],
            #             track_bboxes=track_results['bbox_results'])
            det_bboxes = result['det_bboxes'][0]
            track_bboxes = result['track_bboxes'][0]
            # print("lllllllllllllll",det_bboxes)
            print("lllllllllllllllffffffffffffffffff", track_bboxes)
            result2, max_id = inference_mot(model2, img2, frame_id=i, bboxes1=bboxes2, ids1=ids2, labels1=labels2,
                                            max_id=max_id)
            det_bboxes2 = result2['det_bboxes'][0]
            track_bboxes2 = result2['track_bboxes'][0]

            print("oooooooooooooooooooooooooooooooooo",track_bboxes2)
            ############NMS##########################NMS##########################NMS##########
            # thresh = 0.3
            # # print(len(track_bboxes))
            # track_bboxes = all_nms(track_bboxes, thresh)
            # # print("2", track_bboxes)
            # track_bboxes2 = all_nms(track_bboxes2, thresh)
            # #############NMS##########################NMS##########################NMS##########
            # 遍历两组检测数据并融合信任分配
            # results = []
            # conflicts = []
            # for i in range(detection_data1.shape[0]):
            #    m1 = initialize_mass_function(detection_data1[i, -1])
            #    m2 = initialize_mass_function(detection_data2[i, -1])
            #    combined_m, conflict = combine_mass_functions(m1, m2)
            #    results.append(combined_m)
            #   conflicts.append(conflict)
            #track_bboxes, track_bboxes2 = get_matched_trackboxes(track_bboxes, track_bboxes2)
            #print("trace_bboxes", track_bboxes)
            #print("track_bboxes2", track_bboxes2)
            #track_bboxes = np.array(track_bboxes)
            #track_bboxes2 = np.array(track_bboxes2)
            results = []
            conflicts = []
            image11 = image1.copy()
            image22 = image2.copy()
            threshold = 0.2
            os.makedirs('./imagetrackbox/' + str(dirrr), exist_ok=True)
            for ss in range(track_bboxes.shape[0]):
                for kop in range(track_bboxes2.shape[0]):
                    if track_bboxes[ss,0]==track_bboxes2[kop, 0]:

                        boxes1 = torch.tensor(track_bboxes[:, 1:5], dtype=torch.long)
                        boxes2 = torch.tensor(track_bboxes2[:, 1:5], dtype=torch.long)
                        m1 = initialize_mass_function(track_bboxes[ss, -1])
                        m2 = initialize_mass_function(track_bboxes2[kop, -1])
                        combined_m, conflict = combine_mass_functions(m1, m2)
                        if (combined_m['target'] > threshold):
                            track_bboxes[ss, -1]=combined_m['target']
                            track_bboxes2[kop, -1]=combined_m['target']
                            x1, y1, x2, y2 = boxes1[ss].numpy()
                            cv2.rectangle(image11, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image11, str(combined_m['target'])[0:6], (x1 + 30, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 0, 0), 2)
                            # cv2.putText(image11, str(track_bboxes[ss,0]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # cv2.imwrite("./imagetrackbox/{}/boxesA2-{}.jpg".format(dirrr, i), image11)

                            x1, y1, x2, y2 = boxes2[kop].numpy()
                            cv2.rectangle(image22, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image22, str(combined_m['target'])[0:6], (x1 + 30, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 0, 0), 2)
                            # cv2.putText(image22, str(track_bboxes[ss,0]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # cv2.imwrite("./imagetrackbox/{}/boxesA2-{}.jpg".format(dirrr, i), image11)

                        results.append(combined_m)
                        conflicts.append(conflict)

            # print("kkkkkkkkkkkkkkkkkkkk", conflicts)
            # print("mmmmmmmmmmmmmmmmmmmmm", results)

            track_bboxes111 = track_bboxes
            track_bboxes222 = track_bboxes2
            # print("bbbbbbbbbbbbbbbbbbbbbbbb", track_bboxes111)
            # print("ccccccccccccccccccccccccc", track_bboxes)
            #####################################################################
            bboxes11 = torch.tensor(track_bboxes111[:, 1:5], dtype=torch.long)
            # print("xxxxxxxxxxxxxxxxxxxxxxxx", bboxes11)
            conf1 = track_bboxes111[:, -1]
            image111 = image1.copy()


            # print("qqqqqqqqqqqqqqqqqqqqqqqqqqq", conf1)
            for bbox, conf in zip(bboxes11.numpy(), conf1):
                x1, y1, x2, y2 = bbox[0:4]
                if (conf > threshold):
                    # print("111")
                    cv2.rectangle(image111, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image111, str(conf)[0:6], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            concatenated_image1 = cv2.hconcat([image111, image11])
            # os.makedirs('./imagetrackbox/' + str(dirrr), exist_ok=True)

            cv2.imwrite("./imagetrackbox/{}/boxesA-{}.jpg".format(dirrr, i), concatenated_image1)
            # cv2.imwrite("./imagetrackbox/{}/boxesA1-{}.jpg".format(dirrr, i), image111)
            # cv2.imwrite("./imagetrackbox/{}/boxesA2-{}.jpg".format(dirrr, i), image11)

            bboxes22 = torch.tensor(track_bboxes222[:, 1:5], dtype=torch.long)
            conf2 = track_bboxes222[:, -1]
            # numbers2 = re.findall(r'[\d\.\-]+', conf2)
            image222 = image2.copy()
            # 将提取的字符串数字转换为浮点数
            # conf2 = [float(num) for num in numbers2]

            for bbox, conf in zip(bboxes22.numpy(), conf2):
                x1, y1, x2, y2 = bbox
                if (conf > threshold):
                    cv2.rectangle(image222, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image222, str(conf)[0:6], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            concatenated_image1 = cv2.hconcat([image222, image22])
            os.makedirs('./imagetrackbox/' + str(dirrr), exist_ok=True)
            cv2.imwrite("./imagetrackbox/{}/boxesB-{}.jpg".format(dirrr, i), concatenated_image1)

            print('wjyok')
            # cv2.imwrite("./imagetrackbox/{}/boxesB-{}.jpg".format(dirrr,i), image2)

            track_bboxes, track_bboxes2 = get_chosed_track(track_bboxes, track_bboxes2, 0.2)

            print("gggggggggggggggggggggggggggggggggg", track_bboxes)
            print("track_bboxes2", track_bboxes2)

            #track_bboxes = np.array(track_bboxes)
            #track_bboxes2 = np.array(track_bboxes2)

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()

            # 计算追踪目标中心点，进而计算两机间变换矩阵   计算检测中心点
            # cent_allclass_det, corner_allclass_det = calculate_cent_corner_pst_det(image1,
            #                                                           det_bboxes)  # corner_allclass:ndarray 2n*2        #calculate_cent_corner_pst有更改
            # cent_allclass2_det, corner_allclass2_det = calculate_cent_corner_pst_det(image2,
            #                                                             det_bboxes2)  # cent_allclass:ndarray n*2

            # '''
            # 计算追踪目标中心点，进而计算两机间变换矩阵   计算检测中心点
            cent_allclass, corner_allclass = calculate_cent_corner_pst(image1,
                                                                       track_bboxes)  # corner_allclass:ndarray 2n*2        #calculate_cent_corner_pst有更改
            cent_allclass2, corner_allclass2 = calculate_cent_corner_pst(image2,
                                                                         track_bboxes2)  # cent_allclass:ndarray n*2

            # 第一帧：
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
                # sift 全局匹配（试验）
                # matching(image1, image2, cent_allclass, corner_allclass, 1)
                # '''
                if args.output is not None:
                    if IN_VIDEO or OUT_VIDEO:

                        out_file = osp.join(out_path + str(dirrr), f'{i:06d}.jpg')
                        out_file2 = osp.join(out_path + str(dir2), f'{i:06d}.jpg')
                    else:
                        out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
                        out_file2 = osp.join(out_path2, img.rsplit(os.sep, 1)[-1])
                else:
                    out_file = None
                model.show_result(
                    img,
                    result,
                    score_thr=args.score_thr,
                    show=args.show,
                    wait_time=int(1000. / fps) if fps else 0,
                    out_file=out_file,
                    backend=args.backend)
                model2.show_result(
                    img2,
                    result2,
                    score_thr=args.score_thr,
                    show=args.show,
                    wait_time=int(1000. / fps) if fps else 0,
                    out_file=out_file2,
                    backend=args.backend)
                # '''
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
            # '''
            # 图像融合可视化
            leftImage = image1
            rightImage = image2
            resulttt = cv2.warpPerspective(leftImage, f1, (
                leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
            resulttt.astype(np.float32)
            resulttt = resulttt / 2
            resulttt[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
            cv2.imwrite("./matchingimages/{}/matching{}-1.jpg".format(dirrr, i), resulttt)

            # '''
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
            leftImage = image2
            rightImage = image1
            resulttt = cv2.warpPerspective(leftImage, f2, (
                leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
            resulttt.astype(np.float32)
            resulttt = resulttt / 2
            resulttt[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
            cv2.imwrite("./matchingimages/{}/matching{}-2.jpg".format(dirrr, i), resulttt)

            # 进行ID更改，将能匹配的新目标赋予同ID(选取旧ID作为关联ID)
            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = \
                B_same_target_refresh_same_ID(B_new_ID, B_pts, B_pts_corner, f2, cent_allclass, track_bboxes,
                                              track_bboxes2, matched_ids, det_bboxes, det_bboxes2, image1,
                                              coID_confirme, thres=80)
            ##################3##################3##################3##################3
            # 不是新目标，且没有匹配上的旧目标，在此处再计算一下看有没有能对的上的ID，防止在新目标出现时没对上那么后续就再也对不上的情况（对比试验/消融实验？？？？）
            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
                A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
                = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                  corner_allclass2, A_max_id, B_max_id, coID_confirme)
            # # 不是新目标，且没有匹配上的旧目标，在此处再计算一下看有没有能对的上的ID，防止在新目标出现时没对上那么后续就再也对不上的情况（对比试验/消融实验？？？？）
            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = same_target_refresh_same_ID(
                A_old_not_matched_ids,
                A_old_not_matched_pts,
                A_old_not_matched_pts_corner,
                f1,
                cent_allclass2, track_bboxes,
                track_bboxes2, matched_ids,
                det_bboxes, det_bboxes2,
                image2, coID_confirme, thres=50)

            # ################supplyment###########################supplyment#############supplyment#############supplyment#############supplyment########
            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
                A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
                = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                  corner_allclass2, A_max_id, B_max_id, coID_confirme)
            # for dots in A_old_not_matched_pts:
            #     cv2.circle(image1, (int(dots[0]), int(dots[1])), 5, (36, 23, 112), 9)
            #     cv2.imshow("fksoadf27", image1)
            # cv2.waitKey(10)
            # for dots in B_old_not_matched_pts:
            #     cv2.circle(image2, (int(dots[0]), int(dots[1])), 5, (36, 23, 112), 9)
            #     cv2.imshow("fksoadf28", image2)
            #     cv2.waitKey(10)

            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme, supplement_bbox2 = not_matched_supplement(
                A_old_not_matched_ids,
                A_old_not_matched_pts,
                A_old_not_matched_pts_corner,
                f1,
                cent_allclass2, track_bboxes,
                track_bboxes2, matched_ids,
                det_bboxes, det_bboxes2,
                image2, coID_confirme, supplement_bbox2, thres=50)
            track_bboxes2, track_bboxes, matched_ids, flag, coID_confirme, supplement_bbox = not_matched_supplement(
                B_old_not_matched_ids,
                B_old_not_matched_pts,
                B_old_not_matched_pts_corner,
                f2,
                cent_allclass2, track_bboxes2, track_bboxes,
                matched_ids, det_bboxes2, det_bboxes,
                image1, coID_confirme, supplement_bbox, thres=50)
            # #######################################################supplyment#############supplyment#############supplyment#############supplyment########
            # 
            # 
            # # 计算追踪目标中心点，进而计算两机间变换矩阵
            # cent_allclass, corner_allclass = calculate_cent_corner_pst(image1,
            #                                                            track_bboxes)  # corner_allclass:ndarray 2n*2
            # cent_allclass2, corner_allclass2 = calculate_cent_corner_pst(image2,
            #                                                              track_bboxes2)  # cent_allclass:ndarray n*2
            # A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
            # B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))
            # matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
            # A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
            #     = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
            #                       corner_allclass2, A_max_id, B_max_id)
            # track_bboxes2, track_bboxes, matched_ids, flag = same_target_refresh_same_ID(B_old_not_matched_ids,
            #                                                                              B_old_not_matched_pts,
            #                                                                              B_old_not_matched_pts_corner,
            #                                                                              f2,
            #                                                                              cent_allclass,
            #                                                                              track_bboxes2,
            #                                                                              track_bboxes, matched_ids,
            #                                                                              det_bboxes2, det_bboxes,
            #                                                                              image1, thres=100)
            # # 两个检测框中低置信度的共同目标补充进来：det_bboxes， det_bboxes2

            # max_id = max(A_max_id, B_max_id)
            # track_bboxes, track_bboxes2, matched_ids = \
            #     low_confidence_target_refresh_same_ID(det_bboxes, det_bboxes2, track_bboxes, track_bboxes2, matched_ids, max_id,
            #                                           image1, image2, f1, track_bboxes_old, track_bboxes2_old)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if len(track_bboxes) != 0:
                A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
                B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ############NMS##########################NMS##########################NMS##########
            thresh = 0.3
            # print(len(track_bboxes))
            track_bboxes = all_nms(track_bboxes, thresh)  # 非极大值抑制
            # print("2", track_bboxes)
            track_bboxes2 = all_nms(track_bboxes2, thresh)
            #############NMS##########################NMS##########################NMS##########

            # 更新result
            result['track_bboxes'][0] = track_bboxes
            result2['track_bboxes'][0] = track_bboxes2
            track_bboxes_old = track_bboxes.copy()
            track_bboxes2_old = track_bboxes2.copy()

            # '''
            bboxes1 = torch.tensor(track_bboxes[:, 1:5], dtype=torch.long)
            ids1 = torch.tensor(track_bboxes[:, 0], dtype=torch.long)
            labels1 = torch.zeros_like(torch.tensor(track_bboxes[:, 0]))
            # labels1 = torch.tensor(track_bboxes[:, 0])

            bboxes2 = torch.tensor(track_bboxes2[:, 1:5], dtype=torch.long)
            ids2 = torch.tensor(track_bboxes2[:, 0], dtype=torch.long)
            labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))
            # labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))

            #####################################################################   绘制边界框
            for bbox, id in zip(bboxes1.numpy(), ids1.numpy()):
                x1, y1, x2, y2 = bbox
                # 在图像上绘制边界框
                cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在图像上绘制标签
                cv2.putText(image1, str(id.item()), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for bbox, id in zip(bboxes2.numpy(), ids2.numpy()):
                x1, y1, x2, y2 = bbox
                # 在图像上绘制边界框
                cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在图像上绘制标签
                cv2.putText(image2, str(id.item()), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            os.makedirs('./imagetrackbox/' + str(dirrr), exist_ok=True)
            cv2.imwrite("./imagetrackbox/{}/boxesA2-{}.jpg".format(dirrr, i), image1)
            cv2.imwrite("./imagetrackbox/{}/boxesB2-{}.jpg".format(dirrr, i), image2)

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()

            # ***********************************************
            if len(supplement_bbox) != 0:
                supplement_dict["frame={}".format(i)] = supplement_bbox[:, 0:5].tolist()
            if len(supplement_bbox2) != 0:
                supplement_dict2["frame={}".format(i)] = supplement_bbox2[:, 0:5].tolist()
            # print("result_dict", result_dict)
            # ***********************************************

            # '''
            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path + str(dirrr), f'{i:06d}.jpg')
                    out_file2 = osp.join(out_path + str(dir2), f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
                    out_file2 = osp.join(out_path2, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None

            model.show_result(
                img,
                result,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=args.backend)
            model2.show_result(
                img2,
                result2,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file2,
                backend=args.backend)
            # '''
            prog_bar.update()
        
        
        

        
        time_end = time.time()
        method = args.method
        # method = "NMS-one_carafe_bytetrack_full_mdmt"
        json_dir = "{}/{}/".format(args.result_dir, method)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open("{0}/{1}.json".format(json_dir, sequence1), "w") as f:
            json.dump(result_dict, f, indent=4)
            print("输出文件A写入完成！")
        with open("{0}/{1}.json".format(json_dir, sequence2), "w") as f2:
            json.dump(result_dict2, f2, indent=4)
            print("输出文件B写入完成！")
        with open("{0}/time.txt".format(json_dir), "a") as f3:
            f3.write("{} time consume :{}\n".format(sequence1, time_end - time_start))
            print("输出文件time.txt写入完成！")

        # with open("./json_resultfiles2/supplement_supplement/{0}/{1}.json".format(method, sequence1), "w") as f5:
        #     json.dump(supplement_dict, f5, indent=4)
        #     print("输出文件zengjiaA写入完成！")
        # with open("./json_resultfiles2/supplement_supplement/{0}/{1}.json".format(method, sequence2), "w") as f6:
        #     json.dump(supplement_dict2, f6, indent=4)
        #     print("输出文件zengjiaB写入完成！")

        # '''
        if args.output and OUT_VIDEO:
            print(f'making the output video at {args.output} with a FPS of {fps}')
            mmcv.frames2video(out_path + str(dirrr), outputname, fps=fps, fourcc='mp4v')
            mmcv.frames2video(out_path + str(dir2), output2name, fps=fps, fourcc='mp4v')
            # out_dir.cleanup()
        # '''
    time_end_all = time.time()
    with open("{0}/time.txt".format(json_dir), "a") as f3:
        f3.write("ALL time consume :{}\n".format(time_end_all - time_start_all))

    # # 评估
    # input_dir = "./json_resultfiles2/supplement_supplement/NMS-one_carafe_bytetrack_full_mdmt"
    # output_dir = "./test1"
    # # input_dir = "./json_resultfiles/Firstframe_initialized_bytetrack_autoassign_full_mdmt-private-half/"

    # json_files_dirs = os.listdir(input_dir)

    # for file in json_files_dirs:
    #     #  print(file)
    #     if "txt" in file or "ipynb" in file:
    #         continue
    #     # if file not in ["26-1.json", "26-2.json"]:
    #     #     continue
    #     json_dir = os.path.join(input_dir, file)
    #     sequence = json_dir.split("/")[-1]
    #     sequence = sequence.split(".")[0]
    #     # print(sequence)
    #     # print(json_dir)
    #     n = 0
    #     maxID = 0
    #     with open(json_dir) as f:
    #         load_json = json.load(f)
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)

    #         with open("{0}/{1}.txt".format(output_dir, sequence), "w") as f_txt:
    #             # 两个循环分别遍历字典的键值对
    #             for (key, values) in load_json.items():
    #                 for value in values:
    #                     maxID = max(maxID, value[0])
    #                 #  print(maxID)
    #             while n <= maxID:
    #                 for (key, values) in load_json.items():
    #                     # print(key)
    #                     frame = key.split("=")[-1]
    #                     # 先找ID=1的
    #                     for value in values:
    #                         if value[0] == n:
    #                             string_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(int(frame), int(value[0]),
    #                                                                                          value[1], value[2],
    #                                                                                          value[3] - value[1],
    #                                                                                          value[4] - value[2],
    #                                                                                          int(1),
    #                                                                                          int(1), int(1))
    #                             f_txt.write(string_line)

    #                     # if key == "frame={}".format(n):
    #                     #     print("bigin")

    #                     # for value in values:
    #                     #     # print(value)
    #                     #     frame = key.split("=")[-1]
    #                     #     string_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(frame, value[0], value[1], value[2], value[3], value[4], int(1), int(3), int(1))
    #                     #     f_txt.write(string_line)
    #                 n += 1

    # print("hello")

    # # parser = ArgumentParser()
    # # parser.add_argument('--test_file_dir', default="./demo/txt/MOT/Firstframe_initialized_faster_rcnn_r50_fpn_carafe_1x_full_mdmt/", help="test file directory")
    # # args = parser.parse_args()

    # # 评价指标
    # metrics = list(mm.metrics.motchallenge_metrics)
    # # 导入gt和ts文件
    # gt_files_dir = "./gt_true"
    # ts_files_dir = "./test1"
    # # gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
    # # ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
    # sequences1 = ["26-1", "31-1", "34-1", "48-1", "52-1", "55-1", "56-1", "57-1", "59-1", "61-1", "62-1", "68-1",
    #               "71-1", "73-1"]
    # sequences2 = ["26-2", "31-2", "34-2", "48-2", "52-2", "55-2", "56-2", "57-2", "59-2", "61-2", "62-2", "68-2",
    #               "71-2", "73-2"]
    # frames = [300, 360, 300, 700, 360, 151, 400, 490, 700, 270, 700, 310, 250, 592]

    # idf11_sum = 0
    # mota_sum = 0
    # nn = 0
    # idf11_sum = []
    # mota_sum = []
    # lsdir = os.listdir(ts_files_dir)
    # lsdir.sort()
    # for ts_file_dir in lsdir:
    #     if "-1" in ts_file_dir:
    #         name = ts_file_dir.split(".")[0]
    #         ind_ = sequences1.index(name)
    #         frame = frames[ind_]
    #         # continue
    #     elif "-2" in ts_file_dir:
    #         name = ts_file_dir.split(".")[0]
    #         ind_ = sequences2.index(name)
    #         frame = frames[ind_]
    #         # continue
    #     else:
    #         continue
    #     gt_file = os.path.join(gt_files_dir, ts_file_dir)
    #     ts_file = os.path.join(ts_files_dir, ts_file_dir)
    #     gt = mm.io.loadtxt(gt_file, min_confidence=1)
    #     ts = mm.io.loadtxt(ts_file)
    #     name = os.path.splitext(os.path.basename(ts_file))[0]
    #     # 计算单个acc
    #     acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

    #     mh = mm.metrics.create()
    #     summary = mh.compute(acc, metrics=metrics, name=name)
    #     # print(summary["idf1"])
    #     idf11 = float(summary["idf1"])
    #     mota = float(summary["mota"])
    #     idf11_sum.append(idf11)
    #     mota_sum.append(mota)
    #     nn += 1
    #     print(idf11)
    #     print(mota)
    #     print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    # idf11_averagee = sum(idf11_sum) / nn
    # mota_averagee = sum(mota_sum) / nn
    # Aidf1 = 0
    # Bidf1 = 0
    # for i in range(len(idf11_sum)):
    #     if i % 2 == 0:
    #         Aidf1 += idf11_sum[i]
    #     else:
    #         Bidf1 += idf11_sum[i]

    # Aidf1 = 2 * Aidf1 / nn
    # Bidf1 = 2 * Bidf1 / nn

    # Amota = 0
    # Bmota = 0
    # for i in range(len(mota_sum)):
    #     if i % 2 == 0:
    #         Amota += mota_sum[i]
    #     else:
    #         Bmota += mota_sum[i]

    # Amota = 2 * Amota / nn
    # Bmota = 2 * Bmota / nn


#  print(idf11_averagee)
#  print(mota_averagee)
# print("A idf1:", Aidf1)
#  print("B idf1:", Bidf1)
# print("A mota:", Amota)
#  print("B mota:", Bmota)


if __name__ == '__main__':
    main()
