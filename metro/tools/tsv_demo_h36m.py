# *****Instruction to reproduce Human3.6M tsv files*****
# Please note our preprocess has several dependencies with GraphCMR and Pose2Mesh Github repos
# Also note that it is a long process to reproduce H36M TSV files
# Please follow the steps below:
#
# (1) Clone and install GraphCMR from https://github.com/nkolot/GraphCMR/
# (2) Follow GraphCMR's instruction and download Human3.6M datasets from the official Human3.6M website 
# (3) Follow GraphCMR data preprocessing to generate "h36m_train_protocol1.npz" and "h36m_valid_protocol2.npz" 
#     To generate the two, please check details in https://github.com/nkolot/GraphCMR/blob/master/datasets/preprocess/README.md
# (4) Download pseudo SMPL annotations from https://github.com/hongsukchoi/Pose2Mesh_RELEASE 
# (5) Clone https://github.com/cocodataset/cocoapi and install cocoapi
# (6) Finally, you can run this script to reproduce our TSV files
#
# *****Important Note*****
#
# We use annotations from 
# https://github.com/hongsukchoi/Pose2Mesh_RELEASE 
# and 
# https://github.com/nkolot/GraphCMR/
# If you use the annotations, please consider citing the following papers:
#
# @Inproceedings{kolotouros2019cmr,
#   Title={Convolutional Mesh Regression for Single-Image Human Shape Reconstruction},
#   Author={Kolotouros, Nikos and Pavlakos, Georgios and Daniilidis, Kostas},
#   Booktitle={CVPR},
#   Year={2019}
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
#   |-- H36M_data 
#   |   |-- images
#   |   |   |-- S11_Directions_1.54138969_000001.jpg
#   |   |   |-- S11_Directions_1.54138969_000006.jpg
#   |   |   |-- S11_Directions_1.54138969_000011.jpg
#   |   |   |-- ...
#   |   |   |-- ...
#   |   |-- S1
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |   |-- S5
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |   |-- S6
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |   |-- S7
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |   |-- S8
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |   |-- S9
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |   |-- S11
#   |   |   |-- MyPoseFeatures
#   |   |   |-- MySegmentsMat
#   |   |   |-- Videos
#   |-- POSE2MESH_annotation_folder (all json files are provided by Pose2Mesh, download them from Pose2Mesh Github repo)
#   |   |-- Human36M_subject*_camera.json 
#   |   |-- Human36M_subject*_joint_3d.json
#   |   |-- Human36M_subject*_data.json 
#   |   |-- Human36M_subject*_smpl_param.json
#   |-- GraphCMR (GraphCMR github repo. Please follow their instruction to generate the following *.npz files) 
#   |   |-- datasets
#   |   |   |-- extras
#   |   |   |   |-- h36m_train_protocol1.npz
#   |   |   |   |-- h36m_valid_protocol2.npz

# Please modify the following folder paths to yours
H36M_videos = '/home/keli/H36M_data'
H36M_images = '/home/keli/H36M_data'
POSE2MESH_annotation_folder = '/home/keli/POSE2MESH_annotation_folder'
GRAPHCMR_dataset_path = "/home/keli/GraphCMR/datasets/extras"
OUTPUT_path = "/home/keli/h36m_tsv_reproduce"

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
from metro.modeling._smpl import SMPL
smpl = SMPL().cuda()

import scipy.misc
from collections import defaultdict
from pycocotools.coco import COCO 

tsv_file = "{}/{}.img.tsv"
hw_file = "{}/{}.hw.tsv"
label_file = "{}/{}.label.tsv"
linelist_file = "{}/{}.linelist.tsv"

J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]

pose2mesh_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

graphcmr_joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')

action_name_dict = {'Directions': 2, 'Discussion': 3, 'Eating': 4, 'Greeting': 5, 'Phoning': 6, 'Posing': 7, 'Purchases': 8,
                'Sitting': 9, 'SittingDown': 10, 'Smoking': 11, 'Photo': 12, 'TakingPhoto': 12, 'Waiting': 13, 'Walking': 14, 'WalkDog': 15,'WalkingDog': 15,
                'WalkTogether': 16}
camera_id_dict = {'54138969':  1, '55011271': 2, '58860488': 3, '60457274': 4}


def get_norm_smpl_mesh(gt_3d_joints, smpl_param, cam_param, smpl_model):
    # Note: A large portion of this function is borrowed from Pose2Mesh
    pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
    # smpl parameters (pose: 72 dimension, shape: 10 dimension)
    smpl_pose = torch.FloatTensor(pose).view(-1, 3)
    smpl_shape = torch.FloatTensor(shape).view(1, -1)
    # translation vector from smpl coordinate to h36m world coordinate
    trans = np.array(trans, dtype=np.float32).reshape(3)
    # camera rotation and translation
    R, t = np.array(cam_param['R'],dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],dtype=np.float32).reshape(3)

    # change to mean shape if beta is too far from it
    smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

    # transform world coordinate to camera coordinate
    root_pose = smpl_pose[0, :].numpy()
    angle = np.linalg.norm(root_pose)
    root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
    root_pose = np.dot(R, root_pose)
    axis, angle = transforms3d.axangles.mat2axangle(root_pose)
    root_pose = axis * angle
    smpl_pose[0] = torch.from_numpy(root_pose)

    smpl_pose = smpl_pose.view(1, -1)

    # get mesh and joint coordinates
    smpl_vertices = smpl_model(smpl_pose, smpl_shape)
    smpl_3d_joints = smpl_model.get_h36m_joints(smpl_vertices)

    new_smpl_mesh_coord = smpl_vertices.numpy().astype(np.float32).reshape(-1, 3);
    new_smpl_joint_coord = smpl_3d_joints.numpy().astype(np.float32).reshape(-1, 3)
    
    new_smpl_mesh_coord *= 1000
    # gt_3d_joints[:,:3] *= 1000
    
    h36m_joint = gt_3d_joints.copy()
    h36m_joint[:,:3] = gt_3d_joints[:,:3] - gt_3d_joints[14,:3] # root-relative
    h36m_joint[:,:3] = h36m_joint[:,:3] * 1000

    new_smpl_mesh_coord = torch.from_numpy(new_smpl_mesh_coord)
    h36m_from_smpl = smpl_model.get_h36m_joints(new_smpl_mesh_coord.unsqueeze(0))[0]
    h36m_from_smpl = np.asarray(h36m_from_smpl)

    h36m_joint = h36m_joint[J24_TO_J14,:3]
    reorg_h36m_from_smpl = np.zeros([14,3])
    reorg_h36m_from_smpl = h36m_from_smpl[H36M_J17_TO_J14,:3]
    # translation alignment
    reorg_h36m_from_smpl = reorg_h36m_from_smpl - np.mean(reorg_h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:]
    error = np.sqrt(np.sum((h36m_joint - reorg_h36m_from_smpl)**2,1)).mean()
    # print('fitting error:',error)
    return smpl_pose, smpl_shape, error

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

db_subaction = {}

def subaction_indentify():
    subjects = [1,5,6,7,8,9,11]
    for i in subjects:
        db_subaction[i] = {}
    for i in subjects:
        for act_name in action_name_dict:   
            db_subaction[i][act_name] = []

    for i in subjects:        
        video_path = op.join(H36M_videos,'S'+str(i),'Videos')
        for act_name in action_name_dict:   
            for filename in os.listdir(video_path):
                if act_name in filename:
                    db_subaction[i][act_name].append(filename)

    for i in subjects:
        for act_name in action_name_dict:   
            db_subaction[i][act_name].sort()

    return db_subaction


def pose2mesh_joint_norm(pose2mesh_3djoint_world, R, t):
    joint_cam = world2cam(pose2mesh_3djoint_world, R, t)
    joint_cam = joint_cam - joint_cam[0]
    pose2mesh_S24 = np.zeros([24,4])
    for j_name in graphcmr_joints_name:
        if j_name in pose2mesh_joints_name:
            pose2mesh_j_index = pose2mesh_joints_name.index(j_name)
            graphcmr_j_index = graphcmr_joints_name.index(j_name)
            pose2mesh_S24[graphcmr_j_index,:3] = joint_cam[pose2mesh_j_index,:3]
            pose2mesh_S24[graphcmr_j_index,3] = 1
    return pose2mesh_S24

def pose2mesh_3d_to_2d_joint(pose2mesh_3djoint_world, R, t, f, c):
    joint_cam = world2cam(pose2mesh_3djoint_world, R, t)
    joint_img = cam2pixel(joint_cam, f, c)
    pose2mesh_S24_2D = np.zeros([1,24,3])
    for j_name in graphcmr_joints_name:
        if j_name in pose2mesh_joints_name:
            pose2mesh_j_index = pose2mesh_joints_name.index(j_name)
            graphcmr_j_index = graphcmr_joints_name.index(j_name)
            pose2mesh_S24_2D[0,graphcmr_j_index,:2] = joint_img[pose2mesh_j_index,:2]
            pose2mesh_S24_2D[0,graphcmr_j_index,-1] = 1
    return pose2mesh_S24_2D

def add_Pose2Mesh_smpl_labels(gt_3d_joints, cam_param, joints, smpl_params, subject_id, action_id, frame_id, smpl_mesh_model):
    # Add Pose2Mesh smpl param
    has_smpl = False
    success_match = True
    fail_subaction = []
    set_pose2mesh_3djoint_world = []
    set_pose2mesh_S24 = []
    set_matching_error = []
    set_subaction_hypo = []

    R = cam_param['R']
    t = cam_param['t']
    f = cam_param['focal']
    c = cam_param['princpt']

    # Given meta data (subject id, action id, frame id), we verify whether Pose2Mesh's label match GraphCMR's label
    # We do it by computing the difference of GT 3d joints between Pose2Mesh and GraphCMR
    # We will compute the GT differences for each subject, respectively
    # Check subject #1
    try:
        pose2mesh_3djoint_world_1 = np.array(joints[str(subject_id)][str(action_id)]['1'][str(frame_id)], dtype=np.float32)
        pose2mesh_S24_1 = pose2mesh_joint_norm(pose2mesh_3djoint_world_1, R, t)
        matching_error_1 = np.sum(np.abs(gt_3d_joints[:13,:3]*1000 - pose2mesh_S24_1[:13,:3]))
        set_pose2mesh_3djoint_world.append(pose2mesh_3djoint_world_1)
        set_pose2mesh_S24.append(pose2mesh_S24_1)
        set_matching_error.append(matching_error_1)
        set_subaction_hypo.append(1)
    except KeyError:
        fail_subaction.append(1)
    # Check subject #2
    try:
        pose2mesh_3djoint_world_2 = np.array(joints[str(subject_id)][str(action_id)]['2'][str(frame_id)], dtype=np.float32)
        pose2mesh_S24_2 = pose2mesh_joint_norm(pose2mesh_3djoint_world_2, R, t)
        matching_error_2 = np.sum(np.abs(gt_3d_joints[:13,:3]*1000 - pose2mesh_S24_2[:13,:3]))
        set_pose2mesh_3djoint_world.append(pose2mesh_3djoint_world_2)
        set_pose2mesh_S24.append(pose2mesh_S24_2)
        set_matching_error.append(matching_error_2)
        set_subaction_hypo.append(2)
    except KeyError:
        fail_subaction.append(2)

    
    if len(fail_subaction)==2:
        # if no 3d GT joints matched
        success_match = False
    else:
        # select the best match we have
        mini_matching_error = np.min(set_matching_error)
        subaction_id = set_subaction_hypo[np.argmin(set_matching_error)]
        pose2mesh_3djoint_world = set_pose2mesh_3djoint_world[np.argmin(set_matching_error)]
        pose2mesh_S24 = set_pose2mesh_S24[np.argmin(set_matching_error)]

        # filter with a matching threshold
        if mini_matching_error >50:
            tmp = 's'+str(subject_id)+'_act'+str(action_id)+'_subact'+str(subaction_id)+'_f'+str(frame_id)
            success_match = False

    # check whether Pose2Mesh smpl param exist
    try:
        smpl_param = smpl_params[str(subject_id)][str(action_id)][str(subaction_id)][str(frame_id)]
        smpl_param['gender'] = 'neutral' 
        has_smpl = True
    except KeyError:
        has_smpl = False

    if has_smpl==True and success_match==True:
        # if the data sample has pseudo SMPL param (provided by Pose2Mesh), 
        # we compute SMPL fiting error and get SMPL param in the camera coordinate 
        smpl_cam_pose, smpl_cam_shape, fitting_error24 = get_norm_smpl_mesh(gt_3d_joints, smpl_param, cam_param, smpl_mesh_model)
    else:
        fitting_error24 = 10000000 # empirically set a super large error
        smpl_cam_pose = [0]*72 
        smpl_cam_shape = [0]*10
        smpl_param = {}

    return has_smpl, success_match, pose2mesh_3djoint_world, smpl_param, smpl_cam_pose, smpl_cam_shape, fitting_error24 

def preproc(mode, output_set):
    if mode=='train':
        protocol = 1
    else:
        protocol = 2

    # init SMPL
    smpl_mesh_model = SMPL()
    # tsv structs we need
    rows, rows_label, rows_hw = [], [], []
    # bbox expansion factor
    scaleFactor = 1.2

    # load datasets provided by GraphCMR
    data_npz_path = GRAPHCMR_dataset_path+'/h36m_'+mode+'_protocol'+str(protocol)+'.npz'
    data_npz = np.load(data_npz_path)
    total_data_count = len(data_npz['imgname']) 
    npz_imgname = data_npz['imgname']
    npz_gt_3d_joints = data_npz['S']
    npz_center = data_npz['center']
    npz_scale = data_npz['scale']
    pose2mesh_index = defaultdict(lambda :0)

    # load annotations provided by Pose2Mesh
    subjects = [1,5,6,7,8,9,11]
    db = COCO()
    cameras = {}
    joints = {}
    smpl_params = {}
    db_subaction = subaction_indentify()
    for subject in subjects:
        print('loading ',op.join(POSE2MESH_annotation_folder, 'Human36M_subject' + str(subject) + '_data.json'))
        with open(op.join(POSE2MESH_annotation_folder, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
            annot = json.load(f)

        if len(db.dataset) == 0:
            for k, v in annot.items():
                db.dataset[k] = v
        else:
            for k, v in annot.items():
                db.dataset[k] += v

        # get camera param
        with open(op.join(POSE2MESH_annotation_folder, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
            cameras[str(subject)] = json.load(f)
        # get joint coordinate
        with open(op.join(POSE2MESH_annotation_folder, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
            joints[str(subject)] = json.load(f)
        # get smpl param
        with open(op.join(POSE2MESH_annotation_folder, 'Human36M_subject' + str(subject) + '_smpl_param.json'), 'r') as f:
            smpl_params[str(subject)] = json.load(f)

    db.createIndex()
    match_smpl = 0
    # go over all images
    for img_i in tqdm(range(total_data_count)):
        # Get 3D joint GT from GraphCMR
        gt_3d_joints = npz_gt_3d_joints[img_i]
        imgname = npz_imgname[img_i].decode('UTF-8')

        # Get meta data (action id, subject id, etc)
        act_name = imgname.split('/')[1].split('.')[0].split('_')[1]
        if act_name not in action_name_dict:
            continue
        cam_id = int(camera_id_dict[imgname.split('.')[1].split('_')[0]])
        frame_id = int(imgname.split('.')[1].split('_')[1])-1
        subject_id = int(imgname.split('/')[1].split('_')[0][1])
        subject_str = 'S%d'%(int(imgname.split('/')[1].split('_')[0][1]))
        action_id = action_name_dict[imgname.split('/')[1].split('.')[0].split('_')[1]]

        # Get camera parameter from Pose2Mesh
        cam_param = cameras[str(subject_id)][str(cam_id)]
        R = np.array(cam_param['R'], dtype=np.float32)
        t = np.array(cam_param['t'], dtype=np.float32)
        f = np.array(cam_param['f'], dtype=np.float32)
        c = np.array(cam_param['c'], dtype=np.float32)
        cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c} 

        if mode=='train':
            # We would like to augment training data with Pose2Mesh's pseudo SMPL param 
            # Here, we will try to associate Pose2Mesh's pseudo SMPL param to GraphCMR training data
            # and we will caculate its SMPL fitting error
            has_smpl, success_match, pose2mesh_3djoint_world, smpl_param, smpl_cam_pose, smpl_cam_shape, fitting_error24 = \
                add_Pose2Mesh_smpl_labels(gt_3d_joints, cam_param, joints, smpl_params, subject_id, action_id, frame_id, smpl_mesh_model)
            
            # if fitting error is greater than the threshold, we skip the SMPL param
            if fitting_error24>50: # I empirically set it to 50, and it seems good enough
                has_smpl = False
        else:
            has_smpl = False

        gt_3d_joints = npz_gt_3d_joints[img_i]
        imgname = npz_imgname[img_i].decode('UTF-8')
        center = npz_center[img_i]
        scale = npz_scale[img_i]
        
        # img data processing
        if 'img' in output_set or 'hw' in output_set:
            img_path = op.join(H36M_images,imgname)
            img = cv2.imread(img_path)
            img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

        # keypoints processing
        has_2d_joints = False
        if mode=='train':
            gt_2d_joints_tag = pose2mesh_3d_to_2d_joint(pose2mesh_3djoint_world, R, t, f, c)
            has_2d_joints = True
        else:
            gt_2d_joints_tag = np.zeros((1,24,3))
            has_2d_joints = False

        gt_3d_joints_tag = np.ones((1,24,4))
        gt_3d_joints_tag[0,:,:] = gt_3d_joints

        # smpl: pose and shape
        if mode=='train' and has_smpl == True and success_match==True:
            match_smpl = match_smpl +1
            smpl_pose = smpl_param['pose']
            smpl_shape = smpl_param['shape']
            smpl_trans = smpl_param['trans']
            smpl_fit_3d_pose = smpl_param['fitted_3d_pose']
            smpl_gender = smpl_param['gender']
            smpl_pose_camera_corrd = np.asarray(smpl_cam_pose[0]).tolist()
            smpl_shape_camera_corrd = np.asarray(smpl_cam_shape[0]).tolist()
        else:
            has_smpl = False
            success_match = False
            smpl_pose = [0]*72
            smpl_shape = [0]*10
            smpl_trans = np.zeros((1,3)).tolist()
            smpl_fit_3d_pose = np.zeros((17,3)).tolist()
            smpl_gender = 'neutral'
            smpl_pose_camera_corrd = [0.0]*72
            smpl_shape_camera_corrd = [0.0]*10
                        
        labels = []
        labels.append({"center": center.tolist(), "scale": scale,
            "2d_joints": gt_2d_joints_tag.tolist(), "has_2d_joints": int(has_2d_joints),
            "3d_joints": gt_3d_joints_tag.tolist(), "has_3d_joints": 1,
            "pose": smpl_pose_camera_corrd, "betas": smpl_shape_camera_corrd, "has_smpl": int(has_smpl)*int(success_match) })

        if 'img' in output_set:
            row = [imgname, img_encoded_str]
            rows.append(row)
        if 'hw' in output_set:
            height = img.shape[0]
            width = img.shape[1]
            row_hw = [imgname, json.dumps([{"height":height, "width":width}])]
            rows_hw.append(row_hw)
        if 'label' in output_set:
            row_label = [imgname, json.dumps(labels)]
            rows_label.append(row_label)

    if 'img' in output_set:
        resolved_tsv_file = tsv_file.format(OUTPUT_path,mode)
        tsv_writer(rows, resolved_tsv_file)
        print('save imgs to ', resolved_tsv_file)
    if 'hw' in output_set:
        resolved_hw_file = hw_file.format(OUTPUT_path,mode)
        tsv_writer(rows_hw, resolved_hw_file)
        print('save hw to ', resolved_hw_file)
    if 'label' in output_set:
        resolved_label_file = label_file.format(OUTPUT_path,mode)
        tsv_writer(rows_label, resolved_label_file)
        # generate linelist file
        resolved_linelist_file = linelist_file.format(OUTPUT_path,mode)
        generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)
        print('save labels to ', resolved_label_file)
    
    if mode=='train':
        print('num of successful smpl fitting for training set:', match_smpl)
        print('keli: it should have 305476 successful fits')
    
def main():
    splits = ['valid', 'train']
    output_set = ['label', 'img', 'hw']
    for s in splits:
        preproc(s, output_set)

if __name__ == '__main__':
    main()
