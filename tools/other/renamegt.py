import os

def rename_gt_txt(folder_path):
    # 遍历目录中的所有文件和文件夹
    for root, dirs, files in os.walk(folder_path):
        # 检查当前文件夹是否包含"gt"文件夹
        if "gt" in dirs:
            gt_folder_path = os.path.join(root, "gt")
            # 遍历"gt"文件夹中的所有文件
            for filename in os.listdir(gt_folder_path):
                if filename.endswith(".txt"):
                    old_file_path = os.path.join(gt_folder_path, filename)
                    new_file_path = os.path.join(gt_folder_path, "gt.txt")
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed {old_file_path} to {new_file_path}")

# 指定包含"test" "train" "val"文件夹的父目录路径
dataset_path = "/home/wjy/wujingyi/Dataset/MDMTchallenge/"
folders = ["test", "train", "val"]

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    rename_gt_txt(folder_path)
