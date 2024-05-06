import torch
import pycolmap
import numpy as np
# print(torch.cuda.is_available())
# print(help(pycolmap.SiftExtractionOptions()))

def delet_same_id(ids1_list, ids2_list):
    new_ids1=[]
    new_ids2=[]
    for ss in range(ids1_list.shape[0]):
        if (ids1_list[ss] not in ids2_list):
            new_ids1.append(ids1_list[ss])
    for kop in range(ids2_list.shape[0]):
        if (ids2_list[kop] not in ids1_list):
            new_ids2.append(ids2_list[kop])
    return np.array(new_ids1), np.array(new_ids2)

ids1_list = np.array([1,2,3,4,5])
ids2_list = np.array([1,2,7,8,5,3])

new_ids1, new_ids2 = delet_same_id(ids1_list, ids2_list)
print(new_ids1, '\n', new_ids2)