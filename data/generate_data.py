# @Filename:    generate_data.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/3/22 1:23 PM


from data_loader import CocoDataset
import multiprocessing
from PIL import Image
import numpy as np
from tqdm import tqdm

dir = 'val2017'
root_dir = '../coco2017'
cpu_count = 2
dataset = CocoDataset('../coco2017/annotations/instances_val2017.json', root_dir, dir)
pool = multiprocessing.Pool(processes=cpu_count)

def write_masked_array(img_id):
    ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = dataset.coco.loadAnns(ann_ids)

    img = Image.open(CocoDataset.get_img_path(img_id, dir, root_dir))
    transformed = CocoDataset.get_img_numpy_array(anns, np.array(img), img_id)
    np.save('../coco2017/masked_arrays/{}/{}.npy'.format(dir, img_id), transformed)
    return True

img_ids = dataset.img_ids
mx = len(img_ids)

with pool as p:
    r = list(tqdm(p.imap(write_masked_array, img_ids[:4]), total=4))