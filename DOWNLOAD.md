# Download

## METRO Model Zoo
We provide our METRO models that are trained for 3D human body and 3D hand reconstruction, respectively. 


| Mesh | Description | Filename | Url |
| --- | --- | --- | --- |
| Human body | Model trained on Human3.6M, COCO, MUCO, UP3D, MPII | `metro_h36m_state_dict.bin` | [OneDrive](https://1drv.ms/u/s!AjSbuBGA5p7maYOP-Yr9C8DdwV4?e=ANPczU) |
| Human body | Model further fine-tuned on 3DPW | `metro_3dpw_state_dict.bin` | [OneDrive](https://1drv.ms/u/s!AjSbuBGA5p7mauboKr9sxY0rxBY?e=ATlXLS) |
| Right hand | Model trained on FreiHAND | `metro_hand_state_dict.bin` | [OneDrive](https://1drv.ms/u/s!AjSbuBGA5p7mazQJ3p0sG-CxMaI?e=rm0cbQ) |


## METRO predictions to FreiHAND Leaderboard

| Description | Filename | Url |
| --- | --- | --- |
| METRO | `ckpt200-sc10_rot0-pred.zip` | [OneDrive](https://1drv.ms/u/s!AjSbuBGA5p7mbO6hR40LiT7vWuY?e=5Ez4ox) |
| METRO with test-time augmentation | `ckpt200-multisc-pred.zip` | [OneDrive](https://1drv.ms/u/s!AjSbuBGA5p7mba52EPdaG5HyadU?e=Db0SYe) |

Please place the downloaded files under the `models/metro_release` directory. The data structure should follow the hierarchy as below. 
```
${ROOT}  
|-- models  
|   |-- metro_release
|   |   |-- metro_h36m_state_dict.bin
|   |   |-- metro_3dpw_state_dict.bin
|   |   |-- metro_hand_state_dict.bin
|-- metro 
|-- README.md 
|-- DOWNLOAD.md 
|-- ... 
|-- ... 
```


## SMPL and MANO models
In our codes, we use SMPL and MANO models to generate template mesh and regression targets. To run our codes smoothly, please visit the official websites of SMPL and MANO, and download the models. 

Any use of third-party models of SMPL and MANO are subject to **Software Copyright License for non-commercial scientific research purposes**. See [SMPL-Model License](https://smpl.is.tue.mpg.de/modellicense) and [MANO License](https://mano.is.tue.mpg.de/license) for details.

- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${ROOT}/metro/modeling/data`.
- Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${ROOT}/metro/modeling/data`.
- (Optional) Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl` from [SMPL](https://smpl.is.tue.mpg.de/downloads). Place them at `${ROOT}/metro/modeling/data/.`.

Please put the downloaded files under the `${ROOT}/metro/modeling/data` directory. The data structure should follow the hierarchy below. 
```
${ROOT}  
|-- metro  
|   |-- modeling
|   |   |-- data
|   |   |   |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
|   |   |   |-- basicModel_f_lbs_10_207_0_v1.0.0.pkl
|   |   |   |-- basicModel_m_lbs_10_207_0_v1.0.0.pkl
|   |   |   |-- MANO_RIGHT.pkl
|-- README.md 
|-- DOWNLOAD.md 
|-- ... 
|-- ... 
```
Please check [/metro/modeling/data/README.md](./metro/modeling/data/README.md) for further details.

## ImageNet pre-trained HRNet models
We use pre-trained HRNet as our CNN backbone. Please visit the official website to download the ImageNet pre-trained HRNet models. 

- Download `hrnetv2_w64_imagenet_pretrained.pth` from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
- Download `cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml` from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
- Download `hrnetv2_w40_imagenet_pretrained.pth` from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
- Download `cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml` from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)

Place the downloaded files under the `models/hrnet` directory. Its structure should follow the below hierarchy. 
```
${ROOT}  
|-- models  
|   |-- hrnet
|   |   |-- hrnetv2_w64_imagenet_pretrained.pth
|   |   |-- cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml 
|   |   |-- hrnetv2_w40_imagenet_pretrained.pth
|   |   |-- cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml 
|-- metro 
|-- README.md 
|-- DOWNLOAD.md 
|-- ... 
|-- ... 
```

### Next step
After downloading all the models above, please check [DEMO.md](DEMO.md) for quick demo.

## Download training and evaluation datasets
We provide the pre-parsed data and pseudo ground truth labels, which are provided by open-source project [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE). Annotations are stored in TSV (Tab-Separated-Values) format. We suggest the readers to download the image files from the offical dataset websites.


```bash
wget https://comming-soon.comming-soon/$DATA_NAME.zip
unzip $DATA_NAME.zip -d $DATA_DIR
```
`DATA_NAME` could be `human3.6m`, `coco_smpl`, `muco`, `up3d`, `mpii`, `3dpw`.


The `dataset` directory structure should follow the below hierarchy.
```
${ROOT}  
|-- datasets  
|   |-- Tax-H36m-coco40k-Muco-UP-Mpii  
|   |   |-- train.yaml 
|   |   |-- train.linelist.tsv  
|   |   |-- train.linelist.lineidx
|   |-- human3.6m  
|   |   |-- train.img.tsv 
|   |   |-- train.hw.tsv 
|   |   |-- train.linelist.tsv    
|   |   |-- smpl/train.label.smpl.p1.tsv
|   |   |-- smpl/train.linelist.smpl.p1.tsv
|   |   |-- valid.protocol2.yaml
|   |   |-- valid_protocol2/valid.img.tsv 
|   |   |-- valid_protocol2/valid.hw.tsv  
|   |   |-- valid_protocol2/valid.label.tsv
|   |   |-- valid_protocol2/valid.linelist.tsv
|   |-- coco_smpl  
|   |   |-- train.img.tsv  
|   |   |-- train.hw.tsv   
|   |   |-- smpl/train.label.tsv
|   |   |-- smpl/train.linelist.tsv
|   |-- muco  
|   |   |-- train.img.tsv  
|   |   |-- train.hw.tsv   
|   |   |-- train.label.tsv
|   |   |-- train.linelist.tsv
|   |-- up3d  
|   |   |-- trainval.img.tsv  
|   |   |-- trainval.hw.tsv   
|   |   |-- trainval.label.tsv
|   |   |-- trainval.linelist.tsv
|   |-- mpii  
|   |   |-- train.img.tsv  
|   |   |-- train.hw.tsv   
|   |   |-- train.label.tsv
|   |   |-- train.linelist.tsv
|   |-- 3dpw 
|   |   |-- train.img.tsv  
|   |   |-- train.hw.tsv   
|   |   |-- train.label.tsv
|   |   |-- train.linelist.tsv
|   |   |-- test_has_gender.yaml
|   |   |-- has_gender/test.img.tsv 
|   |   |-- has_gender/test.hw.tsv  
|   |   |-- has_gender/test.label.tsv
|   |   |-- has_gender/test.linelist.tsv
|   |-- freihand
|   |   |-- train.yaml
|   |   |-- train.img.tsv  
|   |   |-- train.hw.tsv   
|   |   |-- train.label.tsv
|   |   |-- train.linelist.tsv
|   |   |-- test.yaml
|   |   |-- test.img.tsv  
|   |   |-- test.hw.tsv   
|   |   |-- test.label.tsv
|   |   |-- test.linelist.tsv
|-- metro
|-- models 
|-- README.md 
|-- DOWNLOAD.md 
|-- ... 
|-- ... 

```
### Next step
After downloading all the datasets above, please check [EXP.md](EXP.md) for scripts to run training and evaluation.


