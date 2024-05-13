
import os
import re
import cv2
import sys
import xml.etree.ElementTree as ET

# VOC格式的文件的存放路径
path_voc = '/home/wjy/wujingyi/Dataset/MDMT/new_xml'
# Yolo格式的标注文件目标路径
path_yolo ='/home/wjy/wujingyi/Dataset/MDMT/gt_true'
# 类名文件的路径
path_class_name ='garbage.names'

# 获取源文件夹下所有后缀为xml格式的文件
re_xml = re.compile(r'.*(xml)')
src_fname_list = os.listdir(path_voc + '/')
xml_fname_list = []
for fname in src_fname_list:
	if re.findall(re_xml, fname):
		xml_fname_list.append(fname)
print(xml_fname_list)

category = {'car':3, 'bus':2, 'person':1, 'bicycle':4}    #MOT中 2: 'person on vehicle'

for xml_fname in xml_fname_list:
	
    # 打开/创建txt格式的文件
    txt_fname = xml_fname.replace('.xml', '.txt')
    print("{} -> {}".format(xml_fname, txt_fname))
    txt_fpath = path_yolo + '/' + txt_fname
    f = open(txt_fpath, 'w', encoding="utf-8")

    # 读取xml文件
    root1 = ET.parse(os.path.join(path_voc, xml_fname)).getroot()  # xml文件根
    track_all = root1.findall('track')
    for track in track_all:
        id_label = int(track.attrib['id'])
        label = int(category[track.attrib['label']])
        boxes = track.findall('box')
        shape = [1080, 1920]
        for box in boxes:
            # print(box.attrib['frame'])
            frame = int(box.attrib['frame'])+1
            xtl = int(box.attrib['xtl'])
            ytl = int(box.attrib['ytl'])
            xbr = int(box.attrib['xbr'])
            ybr = int(box.attrib['ybr'])
            # cv2.rectangle(image1, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
            # cv2.putText(image1, str(id), (xtl, ybr), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            # cv2.imshow("jj", image1)
            # cv2.waitKey(10)
            confidence = 1
            
            f.write("{}".format(frame)+','+"{}".format(id_label)+','+"{}".format(xtl)+','+"{}".format(ytl)+','
					+"{}".format(xbr-xtl)+','+"{}".format(ybr-ytl)+',0,'+"{}".format(label)+',1')
            f.write('\n')
    f.close()




# # VOC格式的文件的存放路径
# path_voc = './images'
# # Yolo格式的标注文件目标路径
# path_yolo ='./yolo'
# # 类名文件的路径
# path_class_name ='garbage.names'

# # 获取源文件夹下所有后缀为xml格式的文件
# re_xml = re.compile(r'.*(xml)')
# src_fname_list = os.listdir(path_voc + '/')
# xml_fname_list = []
# for fname in src_fname_list:
# 	if re.findall(re_xml, fname):
# 		xml_fname_list.append(fname)

# # 读取类名列表
# with open(path_class_name, encoding="utf-8") as f:
# 	class_name_list = f.read().split('\n')
# 	print(class_name_list)

# # XML标签 正则表达式
# # 检索XML格式文件里面的关键信息
# cc=r'<xmin>(.*)</xmin>'
# xxmin=re.compile(r'<xmin>(.*)</xmin>')
# yymin=re.compile(r'<ymin>(.*)</ymin>')
# xxmax=re.compile(r'<xmax>(.*)</xmax>')
# yymax=re.compile(r'<ymax>(.*)</ymax>')
# wwidth=re.compile(r'<width>(.*)</width>')
# hheight=re.compile(r'<height>(.*)</height>')
# nname=re.compile(r'<name>(.*)</name>')


# # 遍历所有的XML文件
# for xml_fname in xml_fname_list:
# 	xml_file = open(path_voc + xml_fname, 'r', encoding="utf-8")
# 	xml_file_content = xml_file.read()
# 	xml_file.close()
	
# 	# 类名列表
# 	name_list = (re.findall(nname, xml_file_content))
	
# 	# 打开/创建txt格式的文件
# 	txt_fname = xml_fname.replace('.xml', '.txt')
# 	print("{} -> {}".format(xml_fname, txt_fname))
# 	txt_fpath = path_yolo + '/' + txt_fname
# 	f = open(txt_fpath, 'w', encoding="utf-8")
# 	# 获取图像尺寸
# 	width=int(re.findall(wwidth,xml_file_content)[0])
# 	height=int(re.findall(hheight,xml_file_content)[0])
# 	# 获取矩形框左上角跟右下角的坐标
# 	for j in range(re.findall(xxmin,xml_file_content).__len__()):
# 		# 类名索引列表
# 		name = name_list[j]
# 		class_id = class_name_list.index(name)
# 		xmin=int(re.findall(xxmin,xml_file_content)[j])
# 		ymin=int(re.findall(yymin,xml_file_content)[j])
# 		xmax=int(re.findall(xxmax,xml_file_content)[j])
# 		ymax=int(re.findall(yymax,xml_file_content)[j])

# 		if j!=0:
# 			f.write('\n')
# 		f.write("{}".format(class_id)+' '+'%0.4f'%((float(xmin)+float(xmax))/float(width)/2)+' '+'%0.4f'%((float(ymin)+float(ymax))/float(height)/2)+' '+'%0.4f'%(float(xmax-xmin)/float(width))+' '+'%0.4f'%(float(ymax-ymin)/float(height)))
# 	f.close()