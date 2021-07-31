import os
import os.path as op
import json
import cv2
import base64
import numpy as np
import code
import torch
from tqdm import tqdm
from metro.utils.tsv_file_ops import tsv_reader, tsv_writer
from metro.utils.tsv_file_ops import generate_linelist_file
from metro.utils.tsv_file_ops import generate_hw_file
from metro.utils.tsv_file import TSVFile
from metro.utils.image_ops import img_from_base64
from metro.modeling._smpl import SMPL
smpl = SMPL().cuda()

from collections import defaultdict
from pycocotools.coco import COCO 

tsv_file = "{}/{}.img.tsv"
hw_file = "{}/{}.hw.tsv"
label_file = "{}/{}.label.tsv"
linelist_file = "{}/{}.linelist.tsv"

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def preproc(dataset_folder, dataset_tsv_folder, split):
    # init SMPL
    smpl_mesh_model = SMPL()

    # bbox expansion factor
    scaleFactor = 1.2

    imgfiles_folder = dataset_folder+'/imageFiles'

    # annotation loading
    rows, rows_label, rows_hw = [], [], []
    db = COCO(op.join(dataset_folder, '3DPW_'+split+'.json'))

    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        img = db.loadImgs(ann['image_id'])[0]
        imgname = op.join(img['sequence'], img['file_name'])
        img_path = op.join(imgfiles_folder, imgname)
        img_data = cv2.imread(img_path)
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img_data)[1])
        width, height = img['width'], img['height']
        bbox = ann['bbox']
        cam_f = img['cam_param']['focal']
        cam_p = img['cam_param']['princpt']

        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        smpl_shape = ann['smpl_param']['shape']
        smpl_pose = ann['smpl_param']['pose']
        gender = ann['smpl_param']['gender']

        smpl_pose_tensor = torch.FloatTensor(smpl_pose).view(1,-1)
        smpl_shape_tensor = torch.FloatTensor(smpl_shape).view(1,-1)
        gt_vertices = smpl_mesh_model(smpl_pose_tensor, smpl_shape_tensor)
        gt_keypoints_3d = smpl_mesh_model.get_joints(gt_vertices) 
        gt_3d_joints = np.asarray(gt_keypoints_3d.cpu())
        gt_3d_joints_tag = np.ones((1,24,4))
        gt_3d_joints_tag[0,:,0:3] = gt_3d_joints
        gt_3d_joints[0] = gt_3d_joints[0] + ann['smpl_param']['trans']
        gt_2d_joints = cam2pixel(gt_3d_joints[0], cam_f, cam_p)
        keypoint_num = gt_2d_joints.shape[0]

        gt_2d_joints_tag = np.ones([24,3])
        gt_2d_joints_tag[:,:2] = gt_2d_joints[:,:2]

        smpl_pose_camera_corrd = np.asarray(smpl_pose_tensor).tolist()[0]
        smpl_shape_camera_corrd = np.asarray(smpl_shape_tensor).tolist()[0]
                        
        labels = []
        labels.append({"center": center, "scale": scale,
            "2d_joints": gt_2d_joints_tag.tolist(), "has_2d_joints": 1,
            "3d_joints": gt_3d_joints_tag.tolist(), "has_3d_joints": 1,
            "gender": gender, "pose": smpl_pose_camera_corrd, "betas": smpl_shape_camera_corrd, "has_smpl": 1 })

        row_label = [imgname, json.dumps(labels)]
        rows_label.append(row_label)
        row = [imgname, img_encoded_str]
        rows.append(row)
        height = img_data.shape[0]
        width = img_data.shape[1]
        row_hw = [imgname, json.dumps([{"height":height, "width":width}])]
        rows_hw.append(row_hw)


    resolved_label_file = label_file.format(dataset_tsv_folder, split)
    print('save to',resolved_label_file)
    tsv_writer(rows_label, resolved_label_file)

    resolved_linelist_file = linelist_file.format(dataset_tsv_folder, split)
    print('save to',resolved_linelist_file)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_tsv_file = tsv_file.format(dataset_tsv_folder, split)
    print('save to',resolved_tsv_file)
    tsv_writer(rows, resolved_tsv_file)

    resolved_hw_file = hw_file.format(dataset_tsv_folder, split)
    print('save to',resolved_hw_file)
    tsv_writer(rows_hw, resolved_hw_file)



def main():
    # *****Instruction to reproduce 3DPW tsv files*****
    #
    # (1) Download 3DPW image files "imageFiles.zip" from the 3DPW websit: https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html
    # (2) Unzip "imageFiles.zip" to get folder "imageFiles"
    # (3) Download pre-parsed 3DPW annotations from https://github.com/hongsukchoi/Pose2Mesh_RELEASE 
    # (4) Clone https://github.com/cocodataset/cocoapi and install cocoapi
    #
    # The final data structure should look like this:
    # ${ROOT}  
    #   |-- datasets 
    #       |-- 3dpw
    #           |-- 3DPW_train.json 
    #           |-- 3DPW_test.json
    #           |-- imageFiles
    #               |-- courtyard_arguing_00
    #               |-- courtyard_backpack_00
    #               |-- ....
    #               |-- ....
    #
    # *****Important Note*****
    #
    # We use annotations from https://github.com/hongsukchoi/Pose2Mesh_RELEASE 
    # If you use the annotations, please consider citing the following paper:
    #
    # @InProceedings{Choi_2020_ECCV_Pose2Mesh,  
    # author = {Choi, Hongsuk and Moon, Gyeongsik and Lee, Kyoung Mu},  
    # title = {Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose},  
    # booktitle = {European Conference on Computer Vision (ECCV)},  
    # year = {2020}  
    # }  

    datasets = ['train','test']
    dataset_img_folder = "./datasets/3dpw"
    dataset_tsv_folder = "./datasets/3dpw_tsv_reproduce"
    for split in datasets:
        preproc(dataset_img_folder, dataset_tsv_folder, split)

if __name__ == '__main__':
    main()
