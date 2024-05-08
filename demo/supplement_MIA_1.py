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
    B_same_target_refresh_same_ID, same_target_refresh_same_ID, get_matched_det
from utils.trans_matrix import supp_compute_transf_matrix as compute_transf_matrix, supp_compute_transf_matrix1_first, \
    supp_compute_transf_matrix1
from utils.supplement import not_matched_supplement, low_confidence_target_refresh_same_ID

#=======在检测上提高置信度=====================

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
           #print("4444444444444333333333333333333")
            # inference process
            max_id = max(A_max_id, B_max_id)
            result, max_id = inference_mot(model, img, frame_id=i, bboxes1=bboxes1, ids1=ids1, labels1=labels1,
                                           max_id=max_id)
            # result = dict(det_bboxes=det_results['bbox_results'],
            #             track_bboxes=track_results['bbox_results'])
            det_bboxes = result['det_bboxes'][0]
            track_bboxes = result['track_bboxes'][0]
            #print("lllllllllllllll", det_bboxes)
            #print("lllllllllllllllffffffffffffffffff", track_bboxes)
            result2, max_id = inference_mot(model2, img2, frame_id=i, bboxes1=bboxes2, ids1=ids2, labels1=labels2,
                                            max_id=max_id)
            det_bboxes2 = result2['det_bboxes'][0]
            track_bboxes2 = result2['track_bboxes'][0]

           # print("oooooooooooooooooooooooooooooooooo", det_bboxes2)
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
            # 计算检测中心点，进而计算两机间变换矩阵   计算检测中心点
            cent_allclass_det, corner_allclass_det = calculate_cent_corner_pst_det(image1,
                                                                       det_bboxes)  # corner_allclass:ndarray 2n*2        #calculate_cent_corner_pst有更改
            cent_allclass2_det, corner_allclass2_det = calculate_cent_corner_pst_det(image2,
                                                                         det_bboxes2)  # cent_allclass:ndarray n*2

            # 第一帧：
            if i == 0:
                f1=supp_compute_transf_matrix1_first(image1, image2, dirrr, i)
                f2=supp_compute_transf_matrix1_first(image2, image1, dirrr, i)
                f1_last=f1
                f2_last=f2

            else:
                f1,f1_last=supp_compute_transf_matrix1(f1_last,image1, image2, dirrr, i)
                f2,f2_last=supp_compute_transf_matrix1(f2_last,image2, image1, dirrr, i)
            #只考虑了f1转换，后面可以增加f2转化
            det_bboxes,det_bboxes2=get_matched_det(cent_allclass_det,corner_allclass_det,det_bboxes,f1,cent_allclass2_det, corner_allclass2_det,det_bboxes2,150)
            det_bboxes = np.array(det_bboxes)
            det_bboxes2 = np.array(det_bboxes2)
            print("bbbbbbbbbbbbbbbbbbbbbbbb",det_bboxes[:, -1])
            print("vvvvvvvvvvvvvvvvvvvvvvvvvv", det_bboxes2[:, -1])
            # 遍历两组检测数据并融合信任分配
            results = []
            conflicts = []
            index = []
            image111=image1.copy()
            image222=image2.copy()
            image11=image1.copy()
            image22=image2.copy()
            threshold = 0.2
            for ss in range(det_bboxes.shape[0]):
                bboxes111 = torch.tensor(det_bboxes[:, 0:4], dtype=torch.long)
                bboxes222 = torch.tensor(det_bboxes2[:, 0:4], dtype=torch.long)
                m1 = initialize_mass_function(det_bboxes[ss, -1])
                m2 = initialize_mass_function(det_bboxes2[ss, -1])
                combined_m, conflict = combine_mass_functions(m1, m2)
                if(combined_m['target']>threshold):
                    x1, y1, x2, y2 = bboxes111[ss].numpy()
                    cv2.rectangle(image111, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image111, str(combined_m['target'])[0:6], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    x1, y1, x2, y2 = bboxes222[ss].numpy()
                    cv2.rectangle(image222, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image222, str(combined_m['target'])[0:6], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                results.append(combined_m)
                conflicts.append(conflict)
                
            print("kkkkkkkkkkkkkkkkkkkk", conflicts)
            print("mmmmmmmmmmmmmmmmmmmmm", results)

            
            #####################################################################
            bboxes11 = torch.tensor(det_bboxes[:, 0:4], dtype=torch.long)
            confs1 = det_bboxes[:, -1]
            for bbox1, conf1 in zip(bboxes11.numpy(), confs1):
                x1, y1, x2, y2 = bbox1
                
                if(float(conf1)> threshold):
                    cv2.rectangle(image11, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image11, str(conf1)[0:6], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            concatenated_image1 = np.concatenate((image11, image111), axis=1)
            os.makedirs('./imagetrackbox/'+str(dirrr), exist_ok=True)
            cv2.imwrite("./imagetrackbox/{}/boxesA-{}.jpg".format(dirrr,i), concatenated_image1)

            bboxes22 = torch.tensor(det_bboxes2[:, 0:4], dtype=torch.long)
            confs2 = det_bboxes2[:, -1]
            for bbox2, conf2 in zip(bboxes22.numpy(), confs2):
                x1, y1, x2, y2 = bbox2
                if(float(conf2) > threshold):
                    cv2.rectangle(image22, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image22, str(conf2)[0:6], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            concatenated_image2 = np.concatenate((image22, image222), axis=1)

            os.makedirs('./imagetrackbox/'+str(dirrr), exist_ok=True)
            cv2.imwrite("./imagetrackbox/{}/boxesB-{}.jpg".format(dirrr,i), concatenated_image2)





            print('wjyok')


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



if __name__ == '__main__':
    main()