import os
import re
import shutil

# def rename_folders(base_path):
#     # 正则表达式匹配类似 '23-1' 格式的文件夹名称
#     pattern = re.compile(r'(\d{2})-1$')

#     # 遍历base_path路径下的所有文件和文件夹
#     for folder_name in os.listdir(base_path):
#         # 构建完整路径
#         folder_path = os.path.join(base_path, folder_name)
        
#         # 检查是否是文件夹
#         if os.path.isdir(folder_path):
#             # 使用正则表达式检查文件夹名称
#             match = pattern.match(folder_name)
#             if match:
#                 # 构建新的文件夹名称，将 '23-1' 改为 '23-2'
#                 new_folder_name = f"{match.group(1)}-2"
#                 new_folder_path = os.path.join(base_path, new_folder_name)

#                 # 重命名文件夹
#                 os.rename(folder_path, new_folder_path)
#                 print(f"Renamed: {folder_path} -> {new_folder_path}")

# # 使用示例
# path = '/home/lab118/pycharm/wjy/mydev/home/wjy/view2in1/'  # 替换为你的实际路径
# rename_folders(path)




# def move_images(source_base_path, target_base_path):
#     # 遍历 source_base_path 路径下的所有文件夹
#     for folder_name in os.listdir(source_base_path):
#         source_folder_path = os.path.join(source_base_path, folder_name, 'img1')
        
#         # 检查是否是文件夹
#         if os.path.isdir(source_folder_path):
#             # 构建目标文件夹路径
#             target_img1_path = os.path.join(target_base_path, folder_name, 'img1')
            
#             # 如果目标路径不存在，则创建它
#             if not os.path.exists(target_img1_path):
#                 os.makedirs(target_img1_path)
            
#             # 遍历当前文件夹中的所有文件
#             for file_name in os.listdir(source_folder_path):
#                 source_file_path = os.path.join(source_folder_path, file_name)
#                 print(source_file_path)
                
#                 # 检查是否是文件以及是否是图像文件（假设图像文件以常见图像格式结尾）
#                 if os.path.isfile(source_file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#                     print(1)
#                     target_file_path = os.path.join(target_img1_path, file_name)
#                     print(target_img1_path)
#                     print(source_file_path)
                    
#                     # 移动文件到目标文件夹
#                     shutil.move(source_file_path, target_file_path)
#                     print(f"Moved: {source_file_path} -> {target_file_path}")

# # 使用示例
# source_path = '/home/lab118/pycharm/wjy/mydev/home/wjy/MDMTchallenge/train'
# target_path = '/home/lab118/pycharm/wjy/mydev/home/wjy/MDMTcross/train'
# move_images(source_path, target_path)


import os
import shutil

def copy_gt_folders(base_path):
    # 遍历 base_path 路径下的所有文件夹
    for folder_name in os.listdir(base_path):
        source_folder_path = os.path.join(base_path, folder_name)
        
        # 检查是否是文件夹并且符合 '23-1' 命名模式
        if os.path.isdir(source_folder_path) and folder_name.endswith('-1'):
            # 构建目标文件夹名称 '23-2'
            target_folder_name = folder_name.replace('-1', '-2')
            target_folder_path = os.path.join(base_path, target_folder_name)
            
            # 检查目标文件夹是否存在
            if os.path.isdir(target_folder_path):
                # 构建源和目标 gt 文件夹路径
                source_gt_path = os.path.join(source_folder_path, 'gt')
                target_gt_path = os.path.join(target_folder_path, 'gt')
                
                # 如果源 gt 文件夹存在，则复制到目标 gt 文件夹
                if os.path.exists(source_gt_path):
                    # 如果目标 gt 文件夹已经存在，则先删除它以避免冲突
                    if os.path.exists(target_gt_path):
                        shutil.rmtree(target_gt_path)
                    
                    # 复制 gt 文件夹
                    shutil.copytree(source_gt_path, target_gt_path)
                    print(f"Copied: {source_gt_path} -> {target_gt_path}")
                else:
                    print(f"Source gt folder does not exist: {source_gt_path}")
            else:
                print(f"Target folder does not exist: {target_folder_path}")

# 使用示例
base_path = '/home/lab118/pycharm/wjy/mydev/home/wjy/MDMTcross/train'
copy_gt_folders(base_path)
