from cv2 import data
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2 as cv
import os

dataDir = os.environ['COCO2017']
dataType = 'train2017'
annFile = os.path.join(dataDir, 'annotations', 'instances_{}.json'.format(dataType))

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories
catIds = coco.getCatIds(catNms=['bicycle','motorcycle'])
imgIds = coco.getImgIds(catIds=catIds )
imgs = coco.loadImgs(imgIds)
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# make target directories
dst_dir = os.path.join(dataDir, 'yolof', dataType)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# determine system command for copying files
if os.name == 'nt':
    copy_cmd = 'COPY {} {}'
else:
    copy_cmd = 'cp {} {}'

for i in range(len(imgs)):
    img = imgs[i]
    file_name = os.path.join(dataDir, dataType, imgs[i]['file_name'])
    dst_file = os.path.join(dst_dir, '{:0>5d}.jpg'.format(i+1))
    # print(copy_cmd.format(file_name, dst_file))
    os.system(copy_cmd.format(file_name, dst_file))
    
