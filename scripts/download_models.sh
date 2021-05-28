# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/models ] ; then
    mkdir -p $REPO_DIR/models
fi
BLOB='https://datarelease.blob.core.windows.net/metro'


# --------------------------------
# Download our pre-trained models
# --------------------------------
if [ ! -d $REPO_DIR/models/metro_release ] ; then
    mkdir -p $REPO_DIR/models/metro_release
fi
# (1) METRO for human mesh reconstruction (trained on H3.6M + COCO + MuCO + UP3D + MPII)
wget -nc $BLOB/models/metro_h36m_state_dict.bin -O $REPO_DIR/models/metro_release/metro_h36m_state_dict.bin
# (2) METRO for human mesh reconstruction (trained on H3.6M + COCO + MuCO + UP3D + MPII, then fine-tuned on 3DPW)
wget -nc $BLOB/models/metro_3dpw_state_dict.bin -O $REPO_DIR/models/metro_release/metro_3dpw_state_dict.bin
# (3) METRO for hand mesh reconstruction (trained on FreiHAND)
wget -nc $BLOB/models/metro_hand_state_dict.bin -O $REPO_DIR/models/metro_release/metro_hand_state_dict.bin


# --------------------------------
# Download the ImageNet pre-trained HRNet models 
# The weights are provided by https://github.com/HRNet/HRNet-Image-Classification
# --------------------------------
if [ ! -d $REPO_DIR/models/hrnet ] ; then
    mkdir -p $REPO_DIR/models/hrnet
fi
wget -nc $BLOB/models/hrnetv2_w64_imagenet_pretrained.pth -O $REPO_DIR/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth
wget -nc $BLOB/models/hrnetv2_w40_imagenet_pretrained.pth -O $REPO_DIR/models/hrnet/hrnetv2_w40_imagenet_pretrained.pth
wget -nc $BLOB/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $REPO_DIR/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
wget -nc $BLOB/models/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $REPO_DIR/models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml


