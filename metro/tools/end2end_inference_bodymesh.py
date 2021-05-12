"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from metro.modeling.bert import BertConfig, METRO
from metro.modeling.bert import METRO_Body_Network as METRO_Network
from metro.modeling._smpl import SMPL, Mesh
from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
import metro.modeling.data.config as cfg

from metro.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text, visualize_reconstruction_and_att_local
from metro.utils.geometric_layers import orthographic_projection
from metro.utils.logger import setup_logger
from metro.utils.miscellaneous import mkdir, set_seed

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

def run_inference(args, image_list, _metro_network, smpl, renderer, mesh_sampler):
    # switch to evaluate mode
    _metro_network.eval()
    
    for image_file in image_list:
        if 'pred' not in image_file:
            att_all = []
            img = Image.open(image_file)
            img_tensor = transform(img)
            img_visual = transform_visualize(img)

            batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
            batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, hidden_states, att = _metro_network(batch_imgs, smpl, mesh_sampler)
                
            # obtain 3d joints from full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # save attantion
            att_max_value = att[-1]
            att_cpu = np.asarray(att_max_value.cpu().detach())
            att_all.append(att_cpu)

            # obtain 3d joints, which are regressed from the full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            # obtain 2d joints, which are projected from 3d joints of smpl mesh
            pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
            pred_2d_431_vertices_from_smpl = orthographic_projection(pred_vertices_sub2, pred_camera)
            visual_imgs_att = visualize_mesh_and_attention( renderer, batch_visual_imgs[0],
                                                        pred_vertices[0].detach(), 
                                                        pred_vertices_sub2[0].detach(), 
                                                        pred_2d_431_vertices_from_smpl[0].detach(),
                                                        pred_2d_joints_from_smpl[0].detach(),
                                                        pred_camera.detach(),
                                                        att[-1][0].detach())

            visual_imgs = visual_imgs_att.transpose(1,2,0)
            visual_imgs = np.asarray(visual_imgs)
                    
            temp_fname = image_file[:-4] + '_metro_pred.jpg'
            print('save to ', temp_fname)
            cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

    return 


def visualize_mesh_and_attention( renderer, images,
                    pred_vertices_full,
                    pred_vertices, 
                    pred_2d_vertices,
                    pred_2d_joints,
                    pred_camera,
                    attention):

    """Tensorboard logging."""
    
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    vertices = pred_vertices.cpu().numpy()
    vertices_2d = pred_2d_vertices.cpu().numpy()
    joints_2d = pred_2d_joints.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    att = attention.cpu().numpy()
    # Visualize reconstruction and attention
    rend_img = visualize_reconstruction_and_att_local(img, 224, vertices_full, vertices, vertices_2d, cam, renderer, joints_2d, att, color='pink')
    rend_img = rend_img.transpose(2,0,1)

    return rend_img


def visualize_mesh_no_text( renderer,
                    images,
                    pred_vertices, 
                    pred_camera):
    """Tensorboard logging."""
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    # Visualize reconstruction only
    rend_img = visualize_reconstruction_no_text(img, 224, vertices, cam, renderer, color='hand')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./test_images/human-body', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")


    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("METRO Inference", args.output_dir, 0)
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()
    # Renderer for visualization
    renderer = Renderer(faces=mesh_smpl.faces.cpu().numpy())
    # Load pretrained model    
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]

            if args.legacy_setting==True:
                # During our paper submission, we were using the original intermediate size, which is 3072 fixed
                # We keep our legacy setting here 
                args.intermediate_size = -1
            else:
                # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
                # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
                args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    logger.info("Run inference")

    image_list = []
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path+'/'+filename) 
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(args, image_list, _metro_network, mesh_smpl, renderer, mesh_sampler)    

if __name__ == "__main__":
    args = parse_args()
    main(args)
