import os
import shutil

# 定义源文件夹路径
source_dir = "../Dataset/MOT17challenge/test"
txt_dir = "/home/wjy/wujingyi/Dataset/MDMT/gt_true/"

# 遍历源文件夹中的每个文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # 新建文件夹det、gt、img1
    det_path = os.path.join(folder_path, "det")
    gt_path = os.path.join(folder_path, "gt")
    img1_path = os.path.join(folder_path, "img1")
    
    os.makedirs(det_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(img1_path, exist_ok=True)
    
    # 移动图片到img1文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            shutil.move(os.path.join(folder_path, filename), os.path.join(img1_path, filename))
    
    # 移动txt文件到gt文件夹
    for txt_file in os.listdir(txt_dir):
        print(txt_file, folder_name)
        if txt_file.startswith(folder_name) and txt_file.endswith(".txt"):
            print(os.path.join(txt_dir, txt_file))
            print(os.path.join(gt_path, txt_file))
            shutil.move(os.path.join(txt_dir, txt_file), os.path.join(gt_path, 'gt.txt'))
