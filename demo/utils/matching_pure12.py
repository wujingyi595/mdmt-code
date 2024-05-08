# _*_ coding: utf-8 _*_
# @Time    :2022/6/17 12:24
# @Author  :LiuZhihao
# @File    :matching_pure.py

# _*_ coding: utf-8 _*_
# @Time    :2022/3/30 19:20
# @Author  :LiuZhihao
# @File    :matching.py.py


# 如果新建的文件名称为test开头，好像运行不了？？？？？？？？？？？？？？？
import os
import cv2
import xml.etree.ElementTree as ET
from mmdet.apis import init_detector, inference_detector
from matplotlib import pyplot as plt
import numpy as np
import mmcv
import csv

def create_sift(leftImage, rightImage):
    # 创造sift
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(leftImage, None)
    kp2, des2 = sift.detectAndCompute(rightImage, None)  # 返回关键点信息和描述符
    print("000000000000000000000000000000000000000000000000000", kp1[4].pt)   # type(kp1[4].pt) ------<class 'tuple'>
    print(kp2, des2)


    # # Convert the training image to RGB
    # training_image = cv2.cvtColor(leftImage, cv2.COLOR_BGR2RGB)
    # # Convert the query image to RGB
    # query_image = cv2.cvtColor(rightImage, cv2.COLOR_BGR2RGB)
    #
    # # Convert the training image to gray scale
    # training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
    # # Convert the query image to gray scale
    # query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    #
    # orb = cv2.ORB_create(200, 2.0)
    # # Find the keypoints in the gray scale training image and compute their ORB descriptor.
    # # The None parameter is needed to indicate that we are not using a mask.
    # kp1, des1 = orb.detectAndCompute(training_gray, None)
    # kp2, des2 = orb.detectAndCompute(query_gray, None)
    #

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # 指定索引树要被遍历的次数

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)   # 得到的matches成对存在，分别为匹配的前两名，每一个包含（distance, queryIdx(下标）, trainIdx（下标）等属性）
    matchesMask = [[0, 0] for i in range(len(matches))]
    # print("matches", matches[0])
    # for i, (m, n) in enumerate(matches):
    #     # if m.distance < 0.07 * n.distance:
    #     if m.distance < 0.7 * n.distance:  # 在多机多目标数据集上0.6会出现误识别，此参数可调！
    #         matchesMask[i] = [1, 0]
    return kp1, kp2, matches, matchesMask

def create_sift1(leftImage, rightImage):
    # 创造sift
    # sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT_create()

    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    use_cuda = torch.cuda.is_available()

    # set torch grad
    torch.set_grad_enabled(False)

    # feature_extractor = Dog(descriptor='sift')#可以更改
    # feature_extractor = Dog(descriptor=args.descriptor.lower())
    if args['descriptor'].lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
        feature_extractor = Dog(descriptor=args['descriptor'].lower())
    elif 'sift' in args['descriptor'].lower():
        feature_extractor = Dog(descriptor='sift')
    elif 'orb' in args['descriptor'].lower():
        feature_extractor = ORBextractor(3000, 1.2, 8)
    elif 'superpoint' in args['descriptor'].lower():
        sp_weights_path = Path(__file__).parent / "extractors/SuperPointPretrainedNetwork/superpoint_v1.pth"
        feature_extractor = SuperPointFrontend(weights_path=sp_weights_path, nms_dist=4, conf_thresh=0.015,
                                               nn_thresh=0.7, cuda=use_cuda)
    elif 'alike' in args['descriptor'].lower():
        feature_extractor = ALike(**alike.configs['alike-l'], device='cuda' if use_cuda else 'cpu', top_k=-1,
                                  scores_th=0.2)
    else:
        raise Exception('Not supported descriptor: "%s".' % args['descriptor'])

    # load json config file
    if "+Boost-" in args['descriptor']:
        # load json config file
        # print("1111111111111")
        config_file = Path(__file__).parent / "config.yaml"
        with open(str(config_file), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # 假设我们想要打印config中对应'descriptor'的配置，如果不存在则返回'Not Found'
        print(config.get(args['descriptor'], 'Not Found'))

        # print(config[args['descriptor']])

        # Model
        feature_booster = FeatureBooster(config.get(args['descriptor']))
        if use_cuda:
            feature_booster.cuda()
        feature_booster.eval()
        # load the model
        model_path = Path(__file__).parent / str("models/" + args['descriptor'] + ".pth")
        print(model_path)
        feature_booster.load_state_dict(torch.load(model_path))

    # print("111111111")

    if 'alike' in args['descriptor'].lower():
        rgb1 = cv2.cvtColor(leftImage, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(rightImage, cv2.COLOR_BGR2RGB)

        pred1 = feature_extractor(rgb1, sub_pixel=True)
        pred2 = feature_extractor(rgb2, sub_pixel=True)

        kp1 = pred1['keypoints']
        kp2 = pred2['keypoints']
        des1 = pred1['descriptors']
        des2 = pred2['descriptors']

        scores1 = pred1['scores']
        scores2 = pred2['scores']

        kp1 = np.hstack((kp1, np.expand_dims(scores1, 1)))
        kp2 = np.hstack((kp2, np.expand_dims(scores2, 1)))

    else:
        leftImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        rightImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

        if 'superpoint' in args['descriptor'].lower():
            leftImage = (leftImage.astype('float32') / 255.)
            rightImage = (rightImage.astype('float32') / 255.)

            kp1, des1, _ = feature_extractor.run(leftImage)
            kp2, des2, _ = feature_extractor.run(rightImage)

            kp1, des1 = kp1.T, des1.T
            kp2, des2 = kp2.T, des2.T

        elif args['descriptor'].lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
            leftImage = (leftImage.astype('float32') / 255.)
            rightImage = (rightImage.astype('float32') / 255.)

            kp1, scores1, des1 = feature_extractor.detectAndCompute(leftImage)
            kp2, scores2, des2 = feature_extractor.detectAndCompute(rightImage)

            kp1 = np.hstack((kp1, np.expand_dims(scores1, 1)))
            kp2 = np.hstack((kp2, np.expand_dims(scores2, 1)))

        elif 'sift' in args['descriptor'].lower():
            leftImage = (leftImage.astype('float32') / 255.)
            rightImage = (rightImage.astype('float32') / 255.)

            kp1, scores1, des1 = feature_extractor.detectAndCompute(leftImage)
            kp2, scores2, des2 = feature_extractor.detectAndCompute(rightImage)

        elif 'orb' in args['descriptor'].lower():
            kps_tuples1, des1 = feature_extractor.detectAndCompute(leftImage)
            kps_tuples2, des2 = feature_extractor.detectAndCompute(rightImage)
            # print(type(des1))

            # des1 = des1.astype(np.float32)
            # des2 = des2.astype(np.float32)
            # print(type(des1))

            # convert keypoints
            kp1 = [cv2.KeyPoint(*kp) for kp in kps_tuples1]
            kp2 = [cv2.KeyPoint(*kp) for kp in kps_tuples2]

            kp1 = np.array(
                [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in kp1],
                dtype=np.float32
            )
            kp2 = np.array(
                [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in kp2],
                dtype=np.float32
            )

    if "+Boost-" in args['descriptor']:
        # boosted the descriptor using trained model
        kp1 = normalize_keypoints(kp1, leftImage.shape)
        kp2 = normalize_keypoints(kp2, rightImage.shape)

        kp1 = torch.from_numpy(kp1.astype(np.float32))
        kp2 = torch.from_numpy(kp2.astype(np.float32))

        if 'orb' in args['descriptor'].lower():
            des1 = np.unpackbits(des1, axis=1, bitorder='little')
            des2 = np.unpackbits(des2, axis=1, bitorder='little')

            des1 = des1 * 2.0 - 1.0
            des2 = des2 * 2.0 - 1.0

        des1 = torch.from_numpy(des1.astype(np.float32))
        des2 = torch.from_numpy(des2.astype(np.float32))

        if use_cuda:
            kp1 = kp1.cuda()

            kp2 = kp2.cuda()

            des1 = des1.cuda()

            des2 = des2.cuda()

        out1 = feature_booster(des1, kp1)
        # print("out1", out1)
        # print(des1.shape, des2.shape, kp1.shape, kp2.shape)
        out2 = feature_booster(des2, kp2)
        kp1 = kp1.cpu().numpy()
        kp2 = kp2.cpu().numpy()
        # print("12334523245432", out1)

        # out2 = feature_booster(des2, kp2)
        if 'boost-b' in args['descriptor'].lower():
            out1 = (out1 >= 0).cpu().detach().numpy()
            out2 = (out2 >= 0).cpu().detach().numpy()

            des1 = np.packbits(out1, axis=1, bitorder='little')
            des2 = np.packbits(out2, axis=1, bitorder='little')

            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)

        else:
            # des1= out.cpu().detach().numpy()
            des1 = out1.cpu().detach().numpy()

            des2 = out2.cpu().detach().numpy()

    if 'orb' in args['descriptor'].lower():
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        ##视情况更改
    # leftImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    # rightImage= cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

    # leftImage = (leftImage.astype('float32') / 255.)
    #  rightImage = (rightImage.astype('float32') / 255.)

    # keypoints1, scores1, des1 = feature_extractor.detectAndCompute(leftImage)
    # keypoints2, scores2, des2 = feature_extractor.detectAndCompute(rightImage)
    #
    # contains_nan = np.isnan(des1).any()
    #  print("popopopopo", "NaN_in_des1", contains_nan )

    # contains_nan = np.isnan(des2).any()
    # print("popopopopo", "NaN_in_des2", contains_nan )

    # kp1 = normalize_keypoints(keypoints1, leftImage.shape)
    # kp2 = normalize_keypoints(keypoints2, rightImage.shape)

    # print("12345", "des1", des1)
    # print("12345", "des2", des2)

    # kp1 = torch.from_numpy(kp1.astype(np.float32))
    # des1 = torch.from_numpy(des1.astype(np.float32))
    #
    # kp2 = normalize_keypoints(keypoints2, rightImage.shape)
    # kp2 = torch.from_numpy(kp2.astype(np.float32))
    # des2 = torch.from_numpy(des2.astype(np.float32))

    # contains_nan = torch.isnan(des1).any()
    # print("Contains NaN:", contains_nan)
    #  print("hhhhhhhhhhhhhhh", "Contains NaN:", contains_nan)

    # contains_nan = torch.isnan(des2).any()
    # print("Contains NaN:", contains_nan)
    # print("hhhhhhhhhhhhhhh", "Contains NaN:", contains_nan)
    # print("67890", "des1", des1)
    # print("67890", "des2", des2)
    # print("000000000000000000000000000000000000",kp1,'\n',des1)
    # print("000000000000000000000000000000000000",kp2,'\n',des2)
    #  if use_cuda:
    #      kp1 = kp1.cuda()

    #     kp2 = kp2.cuda()

    #     des1 = des1.cuda()

    #    des2 = des2.cuda()

    #  contains_nan = torch.isnan(des1).any()
    # print("Contains NaN:", contains_nan)
    #  print("hhhhhhhhhhhhhhh", "Contains NaN:", contains_nan)

    #  contains_nan = torch.isnan(des2).any()
    # print("Contains NaN:", contains_nan)
    # print("hhhhhhhhhhhhhhh", "Contains NaN:", contains_nan)

    # print("mmmmn", des1)

    # print("mmmmn", des2)

    # out1 = feature_booster(des1, kp1)

    # print("12334523245432", out1)

    # out2 = feature_booster(des2, kp2)

    # print("kkkskksksks",out2)

    # print("kkkkd", out1)
    # print("kkkkd", out2)

    # des1 = out1.cpu().detach().numpy()

    # des2 = out2.cpu().detach().numpy()

    # print("14749", "des1", des1)
    # print("14749", "des2", des2)

    # kp1 = kp1.cpu().numpy()
    kp1 = tuple([cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in kp1])
    # kp2 = kp2.cpu().numpy()
    kp2 = tuple([cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in kp2])
    # kp1 = out1.cpu().detach().numpy()
    # kp2 = out2.cpu().detach().numpy()
    # kp1, des1 = sift.detectAndCompute(leftImage, None)
    # kp2, des2 = sift.detectAndCompute(rightImage, None)  # 返回关键点信息和描述符
    # print("000000000000000000000000000000000000", type(kp1[4].pt),kp1[4].pt)   # type(kp1[4].pt) ------<class 'tuple'>
    # print(type(kp2), type(des2),des2)

    # # Convert the training image to RGB
    # training_image = cv2.cvtColor(leftImage, cv2.COLOR_BGR2RGB)
    # # Convert the query image to RGB
    # query_image = cv2.cvtColor(rightImage, cv2.COLOR_BGR2RGB)
    #
    # # Convert the training image to gray scale
    # training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
    # # Convert the query image to gray scale
    # query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    #
    # orb = cv2.ORB_create(200, 2.0)
    # # Find the keypoints in the gray scale training image and compute their ORB descriptor.
    # # The None parameter is needed to indicate that we are not using a mask.
    # kp1, des1 = orb.detectAndCompute(training_gray, None)
    # kp2, des2 = orb.detectAndCompute(query_gray, None)
    #
    # print("des1", des1)
    # print("des2", des2)
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # 指定索引树要被遍历的次数

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)  # 得到的matches成对存在，分别为匹配的前两名，每一个包含（distance, queryIdx(下标）, trainIdx（下标）等属性）
    matchesMask = [[0, 0] for i in range(len(matches))]
    # print("matches", matches[0])
    # for i, (m, n) in enumerate(matches):
    #     # if m.distance < 0.07 * n.distance:
    #     if m.distance < 0.7 * n.distance:  # 在多机多目标数据集上0.6会出现误识别，此参数可调！
    #         matchesMask[i] = [1, 0]
    # des1 = out1.cpu().detach().numpy()
    # des2 = out2.cpu().detach().numpy()
    # print("000000000000000000000000000000000000\n", matches,'\n ',len(matches), matchesMask)   # type(kp1[4].pt) ------<class 'tuple'>
    return kp1, kp2, matches, matchesMask

def compute_matrics(matches, kp1, kp2):
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    good = []
    query_matched = []
    train_matched = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            if m.queryIdx not in query_matched and m.trainIdx not in train_matched:
                query_matched.append(m.queryIdx)
                train_matched.append(m.trainIdx)
                good.append(m)
    # 计算变换矩阵：
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:
        M = 0
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
    return good, matchesMask, M


def compute_matching_points(leftImage, M, pts):
    h, w, channels = leftImage.shape
    # cv2.circle(leftImage, (int(0.530203*w), int(0.247687*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.6994767*w), int(0.343945*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.75781*w),int(0.3990441*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.80594*w),int(0.2731414*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.7552083333333334*w), int(0.7824074074074074*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.43828125*w),    int(0.7384259259259259*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.24557291666666667*w),    int(0.6689814814814815*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(480), int(630)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(40), int(670)), 2, (255, 255, 0), 50)

    dst = cv2.perspectiveTransform(pts, M)

    # for i in range(len(dst)):
    #     # print(dst[i][0][0])
    #     img2 = cv2.circle(rightImage, (int(dst[i][0][0]), int(dst[i][0][1])), 2, (0, 255, 255), 10)
    # # img2 = cv2.polylines(rightImage, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  # 在目标图片外的点会出错
    # plt.imshow(img2)
    # plt.show()
    return dst

def draw_matchingpoints(leftImage, rightImage, kp1, kp2, good, matchesMask, iii):
    # 绘制对应点连线
    drawParams = dict(matchColor=(0, 255, 0), singlePointColor=None,
                      matchesMask=matchesMask, flags=2)  # flag=2只画出匹配点，flag=0把所有的点都画出
    img3 = cv2.drawMatches(leftImage, kp1, rightImage, kp2, good, None, **drawParams)
    # cv2.imshow("new_window:", img3)
    cv2.imwrite("./matching_images/matching{}.jpg".format(iii), img3)

def draw_matchingimg(leftImage, rightImage, M, iii, name):
    result = cv2.warpPerspective(leftImage, M, (leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
    # 融合方法1
    result.astype(np.float32)
    result = result / 2
    result[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
    # result = result/result.max()*255
    cv2.imwrite("./matching_{}{}.jpg".format(name, iii), result)

def matching(leftImage, rightImage):
    kp1, kp2, matches, matchesMask = create_sift(leftImage, rightImage)
    good, matchesMask, M = compute_matrics(matches, kp1, kp2)

    return M


def calculate_cent_corner_pst(img1, result1):
    cent_allclass = []
    corner_allclass = []

    # for calss_num, result in enumerate(result1):
    center_pst = np.array([])
    corner_pst = np.array([])
    for dots in result1:
        # print("dots:", dots)
        x1 = dots[1]
        y1 = dots[2]
        x2 = dots[3]
        y2 = dots[4]
        centx = (x1 + x2) / 2
        centy = (y1 + y2) / 2
        # 收集检测结果的中点和角点
        if center_pst.size == 0:
            center_pst = np.array([[centx, centy]])
        else:
            center_pst = np.append(center_pst, [[centx, centy]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        if corner_pst.size == 0:
            corner_pst = np.array([[x1, y1],
                                   [x2, y2]])
        else:
            corner_pst = np.append(corner_pst, [[x1, y1], [x2, y2]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        # cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 33, 32), 5)
    # center_pst = center_pst.reshape(-1, 2).astype(np.float32)
    # corner_pst = corner_pst.reshape(-1,  2).astype(np.float32)

    # cent_allclass.append(center_pst)
    # corner_allclass.append(corner_pst)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(100)

    return center_pst, corner_pst

#
#
# # 指定模型的配置文件和 checkpoint 文件路径
# # config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# config_file = "work_dirs/swin-t-server/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py"
# # checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# checkpoint_file = "work_dirs/swin-t-server/epoch_7.pth"
# # 根据配置文件和 checkpoint 文件构建模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
#
# if __name__ == '__main__':
#     n = 0
#     data_path = "F:/A_Master_Menu/_A_dataset/MCMOT-new/MDMT/"
#     sequences_path1 = os.path.join(data_path, "1")
#     sequences1 = os.listdir(sequences_path1)
#     sequences_path2 = os.path.join(data_path, "2")
#     sequences2 = os.listdir(sequences_path2)
#     xmldata_path = "F:/A_Master_Menu/_A_dataset/MCMOT-new/xml/"
#     label_file1 = os.path.join(xmldata_path, "1")
#     label_file2 = os.path.join(xmldata_path, "2")
#
#     # sequences1.sort(key= lambda x:int(x.split('-')[0]))
#
#     for s in range(26, len(sequences1)-1):
#         print("s= ",s)
#         sequence1 = os.path.join(sequences_path1, sequences1[s])
#         images1 = os.listdir(sequence1)
#         label_file_r1=os.listdir(label_file1)
#         r1=os.path.join(label_file1, label_file_r1[s])
#         root1 = ET.parse(r1).getroot()   # xml文件根
#
#
#         sequence2 = os.path.join(sequences_path2, sequences2[s])
#         images2 = os.listdir(sequence2)
#         label_file_r2 = os.listdir(label_file2)
#         r2 = os.path.join(label_file2, label_file_r2[s])
#         root2 = ET.parse(r2).getroot()
#         # print("sequence1= ",sequence1)
#         # print("sequence2= ", sequence2)
#         # print("r1= ",r1)
#         # print("r2= ", r2)
#
#
#         numx = 0     #文件1的矩形框总个数
#         numy = 0      #文件2的矩形框总个数
#         # if len(images1)!= len(images2):
#         #     print("!!!!!!!!!!!!!s1!=s2 s1=",sequence1)
#         #     print("!!!!!!!!!!!!!s1!=s2 s2=", sequence2)
#         #     print("len(images1)= ", len(images1))
#         #     print("len(images2)= ", len(images2))
#         # else:
#         #     print("len(images1)= ",len(images1))
#
#
#         num_img= min(len(images1),len(images2))
#         for i in range(num_img):
#
#             if i==5:
#                 break
#             n += 1
#             # i = 0  ##############################################################################################实验需要，临时改动
#             im1 = cv2.imread(os.path.join(sequence1, images1[i]))
#             im2 = cv2.imread(os.path.join(sequence2, images2[i]))
#
#             leftImage = im1.copy()
#             rightImage = im2.copy()
#             # mmdetection检测目标
#             result1 = inference_detector(model, im1)
#             result2 = inference_detector(model, im2)
#             # 得到车辆预测框列表：
#             pred_leftImage = result1[2]  #后续需要剔除置信度低的！！！！！！！！！！！！！！！！！！！！
#             pred_rightImage = result2[2]  #后续需要剔除置信度低的！！！！！！！！！！！！！！！！！！！！
#
#             # 在一个新的窗口中将结果可视化
#             model.show_result(im1, result1, out_file='result-retina2-0412.jpg', score_thr=0.3)    #score_threshold默认0.3
#             model.show_result(im2, result2, out_file='result-retina2-0413.jpg', score_thr=0.3)    #score_threshold默认0.3
#             img1 = cv2.imread("./result-retina2-0412.jpg")
#             img2 = cv2.imread("./result-retina2-0413.jpg")
#
#             center_pst, corner_pst = calculate_cent_corner_pst(img1, result1)
#             # 坐标变换计算结果
#             dst_cent = matching(leftImage, rightImage, center_pst)
#             dst_corner = matching(leftImage, rightImage, corner_pst)
#             # dst_cent = matching(center_pst)
#             # dst_corner = matching(corner_pst)
#
#             if dst_cent is None:
#                 break
#             # print("dist_coener::::::::::::", dst_corner)
#             # print("dst_cent::::::::::::", dst_cent)
#             # 变换后的角点在img2中可视化
#             nn = 0
#             xy1 = (0, 0)
#             xy2 = (0, 0)
#             for xy in dst_corner:
#                 xy = xy.astype(np.int)
#                 if nn == 0:
#                     xy1 = (xy[0, 0], xy[0, 1])
#                     nn += 1
#                 else:
#                     nn = 0
#                     xy2 = (xy[0, 0], xy[0, 1])
#                     cv2.rectangle(img2, xy1, xy2, (299, 3, 4), 1)
#             cv2.imwrite("img2.jpg", img2)
#             cv2.waitKey(100)
#             # 计算im2中inference和坐标变换中目标中点的距离
#             # print("inference results2:", result2[2])
#             # min_points = min(len(result2[2]), len(dst_cent))
#             dist = np.zeros((len(dst_cent), len(result2[2])))
#             valid = []
#             for ii, xy in enumerate(dst_cent):
#                 if xy[0, 0]>1920 or xy[0, 1]>1080:
#                     continue
#                 else:
#                     for j, dots in enumerate(result2[2]):
#                         x1 = dots[0]
#                         y1 = dots[1]
#                         x2 = dots[2]
#                         y2 = dots[3]
#                         centx = int((x1 + x2) / 2)
#                         centy = int((y1 + y2) / 2)
#                         dist[ii, j] = ((xy[0, 0]-centx)**2 + (xy[0, 1]-centy)**2) ** 0.5
#                     # 超参数！35
#                     if min(dist[ii]) > 50:
#                         # valid.append(xy[0])
#                         valid.append([dst_corner[ii*2, 0, 0], dst_corner[ii*2, 0, 1], dst_corner[ii*2+1, 0, 0], dst_corner[ii*2+1, 0, 1]])
#
#             # 可视化补充的检测框
#             # for xyxy in valid:
#             #     xyxy = [int(ii) for ii in xyxy]
#             #     xy1 = (xyxy[0], xyxy[1])
#             #     xy2 = (xyxy[2], xyxy[3])
#             #     cv2.rectangle(img2, xy1, xy2, (2, 2, 222), 4)   #BGR
#             # cv2.imwrite("./matching_images/valid_supply/img{}.jpg".format(n), img2)
#             # print("distence: ", np.min(dist, axis=1))
#             # 补充完检测框之后需要再进行非极大值抑制，此时需要考虑每个预测框的置信度，涉及到补充的检测框的置信度设置问题！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
#             # numpy的数组之间必须是列相同才能append，list不需要相同
#             BB = pred_rightImage.copy().tolist()
#             for app in valid:
#                 BB.append(app)
#
#
#             labels_leftImage = np.array([])
#             labels_rightImage = np.array([])
#             ids_leftImage = []
#             ids_rightImage = []
#             ids_both = []
#             num1 = len(root1)
#             num2 = len(root2)
#             id_with_centxy_dict = {}
#             # 标签获得的匹配点对（中心点，用于计算变换矩阵）
#             pts_src = []
#             pts_dst = []
#             # print(num1, num2)
#             for r in range(max(num1, num2)):
#                 if r < num1:
#                     id1 = root1[r].attrib['id']
#                     label1 = root1[r].attrib['label']
#                     outside1 = int(root1[r][i].attrib['outside'])
#                     occluded1 = int(root1[r][i].attrib['occluded'])
#
#                     x11 = int(root1[r][i].attrib['xtl'])
#                     y11 = int(root1[r][i].attrib['ytl'])
#                     x21 = int(root1[r][i].attrib['xbr'])
#                     y21 = int(root1[r][i].attrib['ybr'])
#                     centx1 = int((x11 + x21) / 2)
#                     centy1 = int((y11 + y21) / 2)
#                     if label1 == 'ignore':
#                         print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn = ", r2)
#                     if outside1 != 1 and (x11 != 0 or y11 != 0) and label1 != 'ignore':
#                         numx += 1
#                         cv2.rectangle(im1, (x11, y11), (x21, y21), (0, 0, 255), 2)
#                         cv2.putText(im1, id1, (x11, y21), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
#                         # 写进labels_leftImage：label框，用于计算精度
#                         if label1 == "car":
#                             if labels_leftImage.size == 0:
#                                 labels_leftImage = np.array([[x11, y11, x21, y21]])
#                                 ids_leftImage.append(id1)
#                             else:
#                                 labels_leftImage = np.append(labels_leftImage, [[x11, y11, x21, y21]], axis=0)
#                                 ids_leftImage.append(id1)
#                             # 写字典-id作为关键词，中心坐标作为值
#                             id_with_centxy_dict["{}".format(id1)] = [centx1, centy1]
#
#
#                 if r < num2:
#                     id2 = root2[r].attrib['id']
#                     label2 = root2[r].attrib['label']
#                     outside2 = int(root2[r][i].attrib['outside'])
#                     occluded2 = int(root2[r][i].attrib['occluded'])
#                     x12 = int(root2[r][i].attrib['xtl'])
#                     y12 = int(root2[r][i].attrib['ytl'])
#                     x22 = int(root2[r][i].attrib['xbr'])
#                     y22 = int(root2[r][i].attrib['ybr'])
#                     centx2 = int((x12 + x22) / 2)
#                     centy2 = int((y12 + y22) / 2)
#                     if outside2 != 1 and (x12 != 0 or y12 != 0) and label2 != 'ignore':
#                         numy += 1
#                         cv2.rectangle(im2, (x12, y12), (x22, y22), (0, 0, 255), 2)
#                         cv2.putText(im2, id2, (x12, y22), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
#                         # 写进labels_rightImage：label框，用于计算精度
#                         if label2 == "car":
#                             if labels_rightImage.size == 0:
#                                 labels_rightImage = np.array([[x12, y12, x22, y22]])
#                                 ids_rightImage.append(id2)
#                                 if id2 in ids_leftImage:
#                                     ids_both.append(id2)
#                             else:
#                                 labels_rightImage = np.append(labels_rightImage, [[x12, y12, x22, y22]], axis=0)
#                                 ids_rightImage.append(id2)
#                                 if id2 in ids_leftImage:
#                                     ids_both.append(id2)
#                                     # 提取字典中值-将12中共同出现的编号他们的中心点值提取出来
#                                     pts_src.append(id_with_centxy_dict["{}".format(id2)])
#                                     pts_dst.append([centx2, centy2])
#
#             # im1 = cv2.resize(im1, (1080, 720))
#             # im2 = cv2.resize(im2, (1080, 720))
#             cv2.imwrite("./matching_images/labelsleft/image{}.jpg".format(n), im1)
#             cv2.imwrite("./matching_images/labelsright/image{}.jpg".format(n), im2)
#             # cv2.waitKey(1)
#
#             # 根据标签中的对应编号目标中心点计算变换矩阵
#             pts_src = np.array(pts_src)
#             pts_dst = np.array(pts_dst)
#             # MIN_MATCH_COUNT》=4
#             MIN_MATCH_COUNT = 5
#             if len(pts_src) > MIN_MATCH_COUNT:
#                 h, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
#                 draw_matchingimg(h, "label_center")
#
#             dst_cent = compute_matching_points(h, center_pst)
#             dst_corner = compute_matching_points(h, corner_pst)
#             if dst_cent is None:
#                 break
#             # print("dist_coener::::::::::::", dst_corner)
#             # print("dst_cent::::::::::::", dst_cent)
#             # 变换后的角点在img2中可视化
#             nn = 0
#             xy1 = (0, 0)
#             xy2 = (0, 0)
#             for xy in dst_corner:
#                 xy = xy.astype(np.int)
#                 if nn == 0:
#                     xy1 = (xy[0, 0], xy[0, 1])
#                     nn += 1
#                 else:
#                     nn = 0
#                     xy2 = (xy[0, 0], xy[0, 1])
#                     cv2.rectangle(img2, xy1, xy2, (299, 3, 4), 1)
#             cv2.imwrite("img2.jpg", img2)
#             cv2.waitKey(100)
#             # 计算im2中inference和坐标变换中目标中点的距离
#             # print("inference results2:", result2[2])
#             # min_points = min(len(result2[2]), len(dst_cent))
#             dist = np.zeros((len(dst_cent), len(result2[2])))
#             valid = []
#             for ii, xy in enumerate(dst_cent):
#                 # if xy[0, 0] > 1920 or xy[0, 1] > 1080:
#                 #     continue
#                 min_x = min(dst_corner[ii * 2, 0, 0], dst_corner[ii * 2 + 1, 0, 0])
#
#                 max_x = max(dst_corner[ii * 2, 0, 0], dst_corner[ii * 2 + 1, 0, 0])
#                 min_y = min(dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 1])
#                 max_y = max(dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 1])
#                 if min_x>1920 or max_x<0 or min_y > 1080 or max_y < 0:
#                     continue
#                 else:
#                     for j, dots in enumerate(result2[2]):
#                         x1 = dots[0]
#                         y1 = dots[1]
#                         x2 = dots[2]
#                         y2 = dots[3]
#                         centx = int((x1 + x2) / 2)
#                         centy = int((y1 + y2) / 2)
#                         dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
#                     # 超参数！35
#                     if min(dist[ii]) > 50:
#                         # valid.append(xy[0])
#                         valid.append([dst_corner[ii * 2, 0, 0], dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 0],
#                                       dst_corner[ii * 2 + 1, 0, 1]])
