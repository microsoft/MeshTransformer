import os
import os.path as op
import json
import cv2
import base64
import sys
import argparse
import numpy as np
import pickle
import code
import imageio
import torch
from tqdm import tqdm
from metro.utils.tsv_file_ops import tsv_reader, tsv_writer
from metro.utils.tsv_file_ops import generate_linelist_file
from metro.utils.tsv_file_ops import generate_hw_file
from metro.utils.tsv_file import TSVFile
from metro.utils.image_ops import img_from_base64
import scipy.misc


# To generate a tsv file:
data_path = "/raid/keli/work/human-mesh/up-3d-tsv/up-3d"
img_list = os.listdir(data_path)
tsv_file = "/raid/keli/work/human-mesh/up-3d-tsv/data/{}.img.tsv"
hw_file = "/raid/keli/work/human-mesh/up-3d-tsv/data/{}.hw.tsv"
linelist_file = "/raid/keli/work/human-mesh/up-3d-tsv/data/{}.linelist.tsv"


def up_3d_extract(dataset_path, out_path, mode):

    # structs we need
    rows, rows_label, rows_hw = [], [], []

    # training splits
    txt_file = os.path.join(dataset_path, 'trainval.txt')

    file = open(txt_file, 'r')
    txt_content = file.read()
    imgs = txt_content.split('\n')

    # go over all images
    for img_i in tqdm(imgs):
        # skip empty row in txt
        if len(img_i) == 0:
            continue

        # image name 
        img_base = img_i[1:-10]
        img_name = '%s_image.png'%img_base
        
        img_path = op.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
        row = [img_name, img_encoded_str]
        rows.append(row)

    resolved_tsv_file = tsv_file.format(mode)
    tsv_writer(rows, resolved_tsv_file)
    


def main():
    datasets = ['trainval']
    dataset_path = "/raid/keli/work/human-mesh/up-3d-tsv/up-3d"
    out_path = "/raid/keli/work/human-mesh/up-3d-tsv/data"
    for split in datasets:
        up_3d_extract(dataset_path, out_path, split)

if __name__ == '__main__':
    main()




