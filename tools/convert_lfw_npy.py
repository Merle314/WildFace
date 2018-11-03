import numpy as np
import lfw
import os
from scipy import misc
import cv2


lfw_pairs = 'data/pairs.txt'
lfw_dir = '/media/lab225/Documents/merle/faceDataSet/lfw/align_112x96'
lfw_file_ext = 'jpg'
lfw_npy_dir = '/media/lab225/Documents/merle/faceDataSet/lfw/align_112x96.npy'
pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)
nrof_images = len(paths) #图片的数量
print(nrof_images)
images = np.zeros((nrof_images, 112, 96, 3), dtype=np.float32)
for index, path in enumerate(paths):
    # img = misc.imread(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    img = (img*1.0-127.5)/128.0
    images[index,:,:,:] = img
np.save(lfw_npy_dir, images)
# images.tofile(lfw_npy_dir)