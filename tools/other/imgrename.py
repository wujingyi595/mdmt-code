import os

def rename_images(folder_path):
    img_folder = os.path.join(folder_path, "img1")
    if not os.path.exists(img_folder):
        print(f"No img1 folder found in {folder_path}")
        return

    image_files = sorted(os.listdir(img_folder))
    print(image_files)
    for i, filename in enumerate(image_files, start=1):
        old_file_path = os.path.join(img_folder, filename)
        new_filename = f"{i:08d}.jpg"
        new_file_path = os.path.join(img_folder, new_filename)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")

# 指定要遍历的目录路径
dataset_path = "/home/wjy/wujingyi/Dataset/MOT17challenge/val"

# 遍历目录中的所有文件夹
for root, dirs, files in os.walk(dataset_path):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        rename_images(folder_path)
