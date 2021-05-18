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

from metro.modeling._smpl import SMPL
smpl = SMPL().cuda()

def preproc(dataset_img_folder, dataset_tsv_folder, split):
    # get image list based on split definition
    txt_file = os.path.join(dataset_img_folder, 'trainval.txt')
    file = open(txt_file, 'r')
    txt_content = file.read()
    imgs = txt_content.split('\n')

    # structs we will output
    rows, rows_label, rows_hw = [], [], []
    tsv_img_file = dataset_tsv_folder + "/{}.img.tsv"
    tsv_hw_file = dataset_tsv_folder + "/{}.hw.tsv"
    tsv_label_file = dataset_tsv_folder + "/{}.label.tsv"
    tsv_linelist_file = dataset_tsv_folder + "/{}.linelist.tsv"

    # iterate all images
    for img_i in tqdm(imgs):
        # skip empty row in txt
        if len(img_i) == 0:
            continue
        
        # =======================================================
        # preprocess tsv_img_file
        # encode image to bytestring, and save it in tsv_img_file
        img_base = img_i[1:-10]
        img_name = '%s_image.png'%img_base
        img_path = op.join(dataset_img_folder, img_name)
        img = cv2.imread(img_path)
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
        row = [img_name, img_encoded_str]
        rows.append(row)

        # =======================================================
        # preprocess tsv_hw_file
        # save image height & width in tsv_hw_file
        height = img.shape[0]
        width = img.shape[1]
        row_hw = [img_name, json.dumps([{"height":height, "width":width}])]
        rows_hw.append(row_hw)

        # =======================================================
        # preprocess tsv_label_file

        # step 1. keypoints processing
        keypoints_file = os.path.join(dataset_img_folder, '%s_joints.npy'%img_base)
        keypoints = np.load(keypoints_file)
        vis = keypoints[2]
        keypoints = keypoints[:2].T
        gt_2d_joints = np.zeros([1,24,3])
        gt_2d_joints[0,:14,:] = np.hstack([keypoints, np.vstack(vis)])

        # step 2. scale and center
        render_name = os.path.join(dataset_img_folder, '%s_render_light.png' % img_base)
        I = imageio.imread(render_name)  # I = scipy.misc.imread(render_name)
        ys, xs = np.where(np.min(I,axis=2)<255)
        bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]

        # step 3. bbox expansion factor
        scaleFactor = 1.2
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

        # step 4. pose and shape
        pkl_file = os.path.join(dataset_img_folder, '%s_body.pkl' % img_base)
        pkl = pickle.load(open(pkl_file, 'rb'), encoding='latin1') 
        pose = pkl['pose']
        betas = pkl['betas']
        pose_tensor = torch.from_numpy(pose).unsqueeze(0).cuda().float()
        betas_tensor = torch.from_numpy(betas).unsqueeze(0).cuda().float()

        # step 5. 3d pose
        gt_vertices = smpl(pose_tensor, betas_tensor) # output shape: torch.Size([1, 6890, 3]) 
        gt_keypoints_3d = smpl.get_joints(gt_vertices) # output shape: torch.Size([1, 24, 3]) 
        gt_3d_joints = np.asarray(gt_keypoints_3d.cpu())
        gt_3d_joints_tag = np.ones((1,24,4))
        gt_3d_joints_tag[0,:,0:3] = gt_3d_joints

        # step 6. save them in tsv_label_file
        labels = []
        labels.append({"center": center, "scale": scale, 
                    "2d_joints": gt_2d_joints.tolist(), "has_2d_joints": 1,
                    "3d_joints": gt_3d_joints_tag.tolist(), "has_3d_joints": 1,
                    "pose": pose.tolist(), "betas": betas.tolist(), "has_smpl": 1 })
        row_label = [img_name, json.dumps(labels)]
        rows_label.append(row_label)

    resolved_tsv_file = tsv_img_file.format(split)
    tsv_writer(rows, resolved_tsv_file)
    resolved_label_file = tsv_label_file.format(split)
    tsv_writer(rows_label, resolved_label_file)
    resolved_tsv_file = tsv_hw_file.format(split)
    tsv_writer(rows_hw, resolved_tsv_file)
    # generate linelist file
    resolved_linelist_file = tsv_linelist_file.format(split)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

def main():
    datasets = ['trainval']
    # download https://files.is.tuebingen.mpg.de/classner/up/datasets/up-3d.zip
    # unzip it and put all files in "./datasets/up-3d"
    dataset_img_folder = "./datasets/up-3d"
    dataset_tsv_folder = "./datasets/up-3d-tsv"
    for split in datasets:
        preproc(dataset_img_folder, dataset_tsv_folder, split)

if __name__ == '__main__':
    main()
