import os

def count_images_in_folder(folder_path):
    """
    计算文件夹中的图片数量
    """
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            count += 1
    return count

def create_seqinfo_file(folder_path, seq_length):
    """
    在指定文件夹中创建seqinfo.ini文件，并写入内容
    """
    seqinfo_path = os.path.join(folder_path, "seqinfo.ini")
    print(seqinfo_path)
    folder_name = os.path.basename(folder_path)
    with open(seqinfo_path, "w") as f:
        f.write("[Sequence]\n")
        f.write("name={}\n".format(folder_name))
        f.write("imDir=img1\n")
        f.write("frameRate=30\n")
        f.write("seqLength={}\n".format(seq_length))
        f.write("imWidth=1920\n")
        f.write("imHeight=1080\n")
        f.write("imExt=.jpg\n")

# 定义源文件夹路径
source_dir = "/home/wjy/wujingyi/Dataset/MOT17challenge/train"

# 遍历源文件夹中的每个文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    img1_path = os.path.join(folder_path, "img1")
    
    # 计算图片数量
    seq_length = count_images_in_folder(img1_path)
    
    # 创建seqinfo.ini文件并写入内容
    create_seqinfo_file(folder_path, seq_length)
