# *****Instruction to reproduce FreiHAND tsv files*****
# Please follow the steps below:
#
# (1) Download FreiHAND dataset (FreiHAND_pub_v2.zip) from https://lmb.informatik.uni-freiburg.de/projects/freihand/ 
#     Note: During the time we developing this project, FreiHAND evaluation GT were not avaiable
#           all our eval results were obtained from FreiHAND eval server
#
# (2) Download additional FreiHAND annotations from https://github.com/hongsukchoi/Pose2Mesh_RELEASE 
# (3) Clone https://github.com/cocodataset/cocoapi and install cocoapi
# (4) Finally, you can run this script to reproduce our TSV files
#
# *****Important Note*****
#
# We use annotations from 
# https://github.com/hongsukchoi/Pose2Mesh_RELEASE 
# and 
# https://lmb.informatik.uni-freiburg.de/projects/freihand/ 
# If you use the annotations, please consider citing the following papers:
#
# @InProceedings{Freihand2019,
#   author    = {Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russel, Max Argus and Thomas Brox},
#   title     = {FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images},
#   booktitle    = {IEEE International Conference on Computer Vision (ICCV)},
#   year      = {2019},
#   url          = {"https://lmb.informatik.uni-freiburg.de/projects/freihand/"}
# }
# @InProceedings{Choi_2020_ECCV_Pose2Mesh,  
# author = {Choi, Hongsuk and Moon, Gyeongsik and Lee, Kyoung Mu},  
# title = {Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose},  
# booktitle = {ECCV},  
# year = {2020}  
# }  

# The final data sturcture should look like this:
# ${HOME}  
#   |-- MeshTransformer (our github repo)
#   |-- FreiHAND  (all files are provided by FreiHand, download them from FreiHand official website)
#   |   |-- *.json
#   |   |-- training
#   |   |   |-- rgb
#   |   |   |   |-- *.jpg
#   |   |   |-- mask
#   |   |   |   |-- *.jpg
#   |   |-- evaluation
#   |   |   |-- rgb
#   |   |   |   |-- *.jpg
#   |-- pose2mesh_data (all json files are provided by Pose2Mesh, download them from Pose2Mesh Github repo)
#   |   |-- freihand_train_data.json 
#   |   |-- freihand_train_coco.json 
#   |   |-- freihand_eval_data.json 
#   |   |-- freihand_eval_coco.json 
#   |   |-- hrnet_output_on_hand_trainset.json
#   |   |-- hrnet_output_on_hand_testset.json
#   |-- hand_repro_tsv  (store the reproduced files)
#   |   |-- outputs
#   |   |   |-- *.tsv


# Please modify the following folder paths to yours
out_path = "/raid/keli/hand_repro_tsv/outputs"
scaleFactor = 1.2 # bbox expansion factor
fhand_img_folder = '/raid/keli/FreiHAND'
eccv_annotation_folder = '/raid/keli/pose2mesh_data'

tsv_file = "{}/{}.img.tsv"
hw_file = "{}/{}.hw.tsv"
label_file = "{}/{}.label.tsv"
linelist_file = "{}/{}.linelist.tsv"


import os
import os.path as op
from os.path import join
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
import transforms3d
from tqdm import tqdm

from metro.utils.tsv_file_ops import tsv_reader, tsv_writer
from metro.utils.tsv_file_ops import generate_linelist_file
from metro.utils.tsv_file_ops import generate_hw_file
from metro.utils.tsv_file import TSVFile
from metro.utils.image_ops import img_from_base64
from metro.modeling._mano import MANO
mano_mesh_model = MANO() # init MANO
import scipy.misc
from collections import defaultdict
from pycocotools.coco import COCO 


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


def get_mano_coord(mesh_model, mano_param, cam_param):
    pose, shape = mano_param['pose'], mano_param['shape']
    # mano parameters (pose: 48 dimension, shape: 10 dimension)
    mano_pose = torch.FloatTensor(pose).view(-1, 3)
    mano_shape = torch.FloatTensor(shape).view(1, -1)
    # camera rotation and translation
    R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'], dtype=np.float32).reshape(3)
    mano_trans = torch.from_numpy(t).view(-1, 3)

    # transform world coordinate to camera coordinate
    root_pose = mano_pose[mesh_model.root_joint_idx, :].numpy()
    angle = np.linalg.norm(root_pose)
    root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
    root_pose = np.dot(R, root_pose)
    axis, angle = transforms3d.axangles.mat2axangle(root_pose)
    root_pose = axis * angle
    mano_pose[mesh_model.root_joint_idx] = torch.from_numpy(root_pose)
    mano_pose = mano_pose.view(1, -1)

    # get mesh and joint coordinates
    mano_mesh_coord, mano_joint_coord = mesh_model.layer(mano_pose, mano_shape, mano_trans)
    mano_mesh_coord = mano_mesh_coord.numpy().reshape(mesh_model.vertex_num, 3);
    mano_joint_coord = mano_joint_coord.numpy().reshape(mesh_model.joint_num, 3)

    return mano_mesh_coord, mano_joint_coord

def process_bbox(bbox, aspect_ratio=None, scale=1.2):
    # sanitize bboxes
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x+(w-1), y+(h-1)
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2, y2])
        return bbox
    else:
        return None

def compute_3d_joints_error(mesh_model, mano_param, cam_param, gt_3d_joints, img_path):
    pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
    # mano parameters (pose: 48 dimension, shape: 10 dimension)
    mano_pose = torch.FloatTensor(pose).view(1, -1)
    mano_shape = torch.FloatTensor(shape).view(1, -1)
    mano_trans = torch.FloatTensor(trans).view(1, -1)

    mano_mesh_coord, mano_joint_coord = mesh_model.layer(mano_pose, mano_shape, mano_trans)
    mano_mesh_coord = mano_mesh_coord.numpy().reshape(mesh_model.vertex_num, 3);
    mano_joint_coord = mano_joint_coord.numpy().reshape(mesh_model.joint_num, 3)
    
    gt_2d_joints = cam2pixel(gt_3d_joints, cam_param['focal'], cam_param['princpt'])
    # find root location
    pred_root = mano_joint_coord[mesh_model.root_joint_idx, :]
    gt_root = gt_3d_joints[mesh_model.root_joint_idx, :]

    # normalize
    mano_mesh_coord = mano_mesh_coord - pred_root[None, :]
    mano_joint_coord_norm = mano_joint_coord - pred_root[None, :]
    gt_3d_joints_norm = gt_3d_joints - gt_root[None, :]
    error = np.sqrt(np.sum((mano_joint_coord_norm - gt_3d_joints_norm) ** 2, 1)).mean()

    return error, None, gt_2d_joints, gt_3d_joints_norm


def extract(data_split):

    if data_split == 'train':
        db = COCO(op.join(eccv_annotation_folder, 'freihand_train_coco.json'))
        with open(op.join(eccv_annotation_folder, 'freihand_train_data.json')) as f:
            data = json.load(f)
        with open(op.join(eccv_annotation_folder, 'hrnet_output_on_hand_trainset.json')) as f:
            hrnet_pred = json.load(f)
    else:
        db = COCO(op.join(eccv_annotation_folder, 'freihand_eval_coco.json'))
        with open(op.join(eccv_annotation_folder, 'freihand_eval_data.json')) as f:
            data = json.load(f)
        with open(op.join(eccv_annotation_folder, 'hrnet_output_on_hand_testset.json')) as f:
            hrnet_pred = json.load(f)

    datalist = []
    k = 0
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']

        hrnet_2dpose_output = hrnet_pred[k]
        k = k + 1
        if hrnet_2dpose_output['image_id']==image_id:
            hrnet_keypoint2d = hrnet_2dpose_output['keypoints']

            img = db.loadImgs(image_id)[0]
            img_path = op.join(fhand_img_folder, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if data_split == 'train':
                cam_param, mano_param, joint_cam = data[db_idx]['cam_param'], data[db_idx]['mano_param'], data[db_idx][
                    'joint_3d']
                joint_cam = np.array(joint_cam).reshape(-1, 3)
                bbox = process_bbox(np.array(ann['bbox']))
                if bbox is None: continue

            else:
                # since eval GTs are not available, we put dummy GTs here
                # we will submit our predictions to FreiHAND eval server to get scores 
                cam_param, scale = data[db_idx]['cam_param'], data[db_idx]['scale']
                joint_cam = np.ones((mano_mesh_model.joint_num, 3), dtype=np.float32)  # dummy
                mano_param = {'pose': np.ones((48), dtype=np.float32), 'shape': np.ones((10), dtype=np.float32)}
                bbox = process_bbox(np.array(ann['bbox']))
                if bbox is None: continue

            cam_param['R'] = np.eye(3).astype(np.float32).tolist();
            cam_param['t'] = np.zeros((3), dtype=np.float32)  # dummy

            datalist.append({
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'hrnet_keypoint2d':hrnet_keypoint2d})

    sorted(datalist, key=lambda d: d['img_id'])

    print('total mesh from eccv data:',len(datalist))

    # structs we need
    rows, rows_label, rows_hw = [], [], []

    error_list = []
    for i in tqdm(range(len(datalist))):
        labels = []

        data = datalist[i]
        img_id = data['img_id']
        img_path, img_shape, smpl_param, cam_param = data['img_path'], data['img_shape'], data['mano_param'], data['cam_param']
        imgname = img_path.split('/')[-1]
        gt_3d_joints = data['joint_cam']
        hrnet_keypoint2d = data['hrnet_keypoint2d']
        print(img_path)
        img = cv2.imread(img_path)
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

        smpl_pose_camera_corrd = smpl_param['pose']
        smpl_shape_camera_corrd = smpl_param['shape']

        bbox = data['bbox']
        center = [112,112]
        scaleFactor = 1.2
        scale =  0.9

        if data_split=='train':
            error, img_data, gt_2d_joints, gt_3d_joints_norm = compute_3d_joints_error(mano_mesh_model, smpl_param, cam_param, gt_3d_joints, img_path)
            error_list.append(error)
            # fname = 'vis_hand_joints_%02d.jpg'%(i)
            gt_2d_joints_tag = np.ones([21,3])
            gt_2d_joints_tag[:,:2] = gt_2d_joints[:,:2]
            gt_3d_joints_tag = np.ones([21,4])
            gt_3d_joints_tag[:,:3] = gt_3d_joints_norm[:,:3]

            labels.append({"center": center, "scale": scale,
                "2d_joints": gt_2d_joints_tag.tolist(), "has_2d_joints": 1,
                "3d_joints": gt_3d_joints_tag.tolist(), "has_3d_joints": 1,
                "hrnet_2d_joints": hrnet_keypoint2d, "has_hrnet_2d_joints": 1,
                "pose": smpl_pose_camera_corrd, "betas": smpl_shape_camera_corrd, "has_smpl": 1 })
        else:
            gt_2d_joints_tag = np.zeros([21,3])
            gt_3d_joints_tag = np.zeros([21,4])

            labels.append({"center": center, "scale": scale,
                "2d_joints": gt_2d_joints_tag.tolist(), "has_2d_joints": 0,
                "3d_joints": gt_3d_joints_tag.tolist(), "has_3d_joints": 0,
                "hrnet_2d_joints": hrnet_keypoint2d, "has_hrnet_2d_joints": 1,
                "pose": smpl_pose_camera_corrd.tolist(), "betas": smpl_shape_camera_corrd.tolist(), "has_smpl": 1 })

        row_label = [imgname, json.dumps(labels)]
        rows_label.append(row_label)

        row = [imgname, img_encoded_str]
        rows.append(row)

        height = img.shape[0]
        width = img.shape[1]
        row_hw = [imgname, json.dumps([{"height":height, "width":width}])]
        rows_hw.append(row_hw)


    resolved_label_file = label_file.format(out_path,data_split)
    print('save to',resolved_label_file)
    tsv_writer(rows_label, resolved_label_file)

    # generate linelist file
    resolved_linelist_file = linelist_file.format(out_path,data_split)
    print('save to',resolved_linelist_file)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_tsv_file = tsv_file.format(out_path,data_split)
    print('save to',resolved_tsv_file)
    tsv_writer(rows, resolved_tsv_file)

    resolved_hw_file = hw_file.format(out_path,data_split)
    print('save to',resolved_hw_file)
    tsv_writer(rows_hw, resolved_hw_file)


def main():
    split = ['test','train']
    for s in split:
        extract(s)

if __name__ == '__main__':
    main()
