# Training and evaluation 

In our experiments, we conduct single-node distributed training using a machine with 8 V100 GPUs. 


## 3D hand reconstruction from a single image

### Training

We use the following script to train on FreiHAND dataset. 

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
       metro/tools/run_metro_handmesh.py \
       --train_yaml freihand/train.yaml \
       --val_yaml freihand/test.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 32 \
       --per_gpu_eval_batch_size 32 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,64 
```

Example training log can be found here [freihand_reproduce_log.txt](https://datarelease.blob.core.windows.net/metro/models/freihand_reproduce_log.txt)

### Testing

After training, we use the final checkpoint (trained at 200 epoch) for testing.

We use the following script to generate predictions. It will generate a prediction file called `ckpt200-sc10_rot0-pred.zip`. Afte that, please submit the prediction file to [FreiHAND Leaderboard](https://competitions.codalab.org/competitions/21238) to obtain the evlauation scores.

```bash
python metro/tools/run_hand_inference.py \
       --model_path models/metro_released/metro_hand_state_dict.bin \
```

To reproduce our results, we released our prediction file `ckpt200-sc10_rot0-pred.zip` (see `docs/DOWNLOAD.md`). Please submit the file to the Leaderboard, and you should be able to generate the following results. 

```bash
Evaluation 3D KP results:
auc=0.000, mean_kp3d_avg=71.49 cm
Evaluation 3D KP ALIGNED results:
auc=0.869, mean_kp3d_avg=0.66 cm

Evaluation 3D MESH results:
auc=0.000, mean_kp3d_avg=71.49 cm
Evaluation 3D MESH ALIGNED results:
auc=0.866, mean_kp3d_avg=0.67 cm

F-scores
F@5.0mm = 0.000 	F_aligned@5.0mm = 0.721
F@15.0mm = 0.000 	F_aligned@15.0mm = 0.981
```
Note that our method predicts relative coordinates (there is no global alignment). 
Therefore, only the **aligned** scores are meaningful in our case.


### Test-time augmentation for FreiHAND

In the following script, we perform test-time augmentation to improve the perfomrance on FreiHAND experiments. We will generate a prediction file `ckpt200-multisc-pred.zip`. 

```bash
python metro/tools/run_hand_inference.py \
       --model_path models/metro_released/metro_hand_state_dict.bin \
       --multiscale_inference
```

To reproduce our results, we have released our prediction file `ckpt200-multisc-pred.zip` (see `docs/DOWNLOAD.md`). You may want to submit it to the Leaderboard, and it should produce the following results. 

```bash
Evaluation 3D KP results:
auc=0.000, mean_kp3d_avg=71.49 cm
Evaluation 3D KP ALIGNED results:
auc=0.876, mean_kp3d_avg=0.62 cm

Evaluation 3D MESH results:
auc=0.000, mean_kp3d_avg=71.49 cm
Evaluation 3D MESH ALIGNED results:
auc=0.874, mean_kp3d_avg=0.63 cm

F-scores
F@5.0mm = 0.000 	F_aligned@5.0mm = 0.743
F@15.0mm = 0.000 	F_aligned@15.0mm = 0.984
```

## Human mesh reconstruction from a single image


### Training

We conduct large-scale training on multiple 2D and 3D datasets, including Human3.6M, COCO, MUCO, UP3D, MPII. During training, it will evaluate the performance per epoch, and save the best checkpoints.

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
       metro/tools/run_metro_bodymesh.py \
       --train_yaml Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml human3.6m/valid.protocol2.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 30 \
       --per_gpu_eval_batch_size 30 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,128 
```

Example training log can be found here [h36m_reproduce_log.txt](https://datarelease.blob.core.windows.net/metro/models/h36m_reproduce_log.txt)

### Evaluation on Human3.6M

In the following script, we evaluate our model `metro_h36m_state_dict.bin` on Human3.6M validation set. Check `docs/DOWNLOAD.md` for more details about downloading the model file.

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
          metro/tools/run_metro_bodymesh.py \
          --val_yaml human3.6m/valid.protocol2.yaml \
          --arch hrnet-w64 \
          --num_workers 4 \
          --per_gpu_eval_batch_size 30 \
          --num_hidden_layers 4 \
          --num_attention_heads 4 \
          --input_feat_dim 2051,512,128 \
          --hidden_feat_dim 1024,256,128 \
          --run_eval_only \
          --resume_checkpoint ./models/metro_release/metro_h36m_state_dict.bin 
```

We show the example outputs of this script as below. 
```bash
...
...
...
METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
METRO INFO: Update config parameter hidden_size: 768 -> 1024
METRO INFO: Update config parameter num_attention_heads: 12 -> 4
METRO INFO: Init model from scratch.
METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
METRO INFO: Update config parameter hidden_size: 768 -> 256
METRO INFO: Update config parameter num_attention_heads: 12 -> 4
METRO INFO: Init model from scratch.
METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
METRO INFO: Update config parameter hidden_size: 768 -> 128
METRO INFO: Update config parameter num_attention_heads: 12 -> 4
METRO INFO: Init model from scratch.
METRO INFO: => loading hrnet-v2-w64 model
METRO INFO: Transformers total parameters: 102256646
METRO INFO: Backbone total parameters: 128059944
METRO INFO: Loading state dict from checkpoint metro_release/metro_h36m_state_dict.bin
...
...
...
INFO:METRO:Validation epoch: 0  mPVE:   0.00, mPJPE:  54.04, PAmPJPE:  36.75 
The experiment completed successfully. Finalizing run...
```
 


### Training with 3DPW dataset

We follow prior works that also use 3DPW training data. In order to make the training faster, we **fine-tune** our pre-trained model (`metro_h36m_state_dict.bin`) on 3DPW training set. 

We use the following script for fine-tuning. During fine-tuning, it will evaluate the performance per epoch, and save the best checkpoints. 

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
       metro/tools/run_metro_bodymesh.py \
       --train_yaml 3dpw/train.yaml \
       --val_yaml 3dpw/test_has_gender.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 30 \
       --per_gpu_eval_batch_size 30 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 5 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,128 \
       --resume_checkpoint  {YOUR_PATH/state_dict.bin} \
```


### Evaluation on 3DPW
In the following script, we evaluate our model `metro_3dpw_state_dict.bin` on 3DPW test set. Check `docs/DOWNLOAD.md` for more details about downloading the model file.


```bash
python -m torch.distributed.launch --nproc_per_node=8 \
          metro/tools/run_metro_bodymesh.py \
          --val_yaml 3dpw/test.yaml \
          --arch hrnet-w64 \
          --num_workers 4 \
          --per_gpu_eval_batch_size 30 \
          --num_hidden_layers 4 \
          --num_attention_heads 4 \
          --input_feat_dim 2051,512,128 \
          --hidden_feat_dim 1024,256,128 \
          --run_eval_only \
          --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin 
```

After evaluation, it should reproduce the results below
```bash
...
...
...
METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
METRO INFO: Update config parameter hidden_size: 768 -> 1024
METRO INFO: Update config parameter num_attention_heads: 12 -> 4
METRO INFO: Init model from scratch.
METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
METRO INFO: Update config parameter hidden_size: 768 -> 256
METRO INFO: Update config parameter num_attention_heads: 12 -> 4
METRO INFO: Init model from scratch.
METRO INFO: Update config parameter num_hidden_layers: 12 -> 4
METRO INFO: Update config parameter hidden_size: 768 -> 128
METRO INFO: Update config parameter num_attention_heads: 12 -> 4
METRO INFO: Init model from scratch.
METRO INFO: => loading hrnet-v2-w64 model
METRO INFO: Transformers total parameters: 102256646
METRO INFO: Backbone total parameters: 128059944
METRO INFO: Loading state dict from checkpoint metro_release/metro_3dpw_state_dict.bin
...
...
...
INFO:METRO:Validation epoch: 0  mPVE:  88.28, mPJPE:  77.10, PAmPJPE:  47.90
The experiment completed successfully. Finalizing run...
```

