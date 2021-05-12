# Quick Demo 
We provide demo codes for end-to-end inference here.

Our inference codes will iterate all images in a given folder, and generate the results.

## Human Body Reconstruction 

This demo runs 3D human mesh reconstruction from a single image. 

Our codes require the input images that are already **cropped with the person centered** in the image. The input images should have the size of `224x224`. To run the demo, please place your test images under `./samples/human-body`, and then run the following script.


```bash
python ./metro/tools/end2end_inference_bodymesh.py 
       --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
       --image_file_or_path ./samples/human-body
```
After running, it will generate the results in the folder `./samples/human-body`

 <img src="../samples/human-body/3dpw_test1_metro_pred.jpg" width="650"> 

## Hand Reconstruction 

This demo runs 3D hand reconstruction from a single image. 

You may want to provide the images that are already **cropped with the right-hand centered** in the image. The input images should have the size of `224x224`. Please place the images under `./samples/hand`, and run the following script.

```bash
python ./metro/tools/end2end_inference_handmesh.py 
       --resume_checkpoint ./models/metro_release/metro_hand_state_dict.bin
       --image_file_or_path ./samples/hand
```
After running, it will outputs the results in the folder `./samples/hand`

 <img src="../samples/hand/freihand_sample1_metro_pred.jpg" width="650"> 

## Limitations

 - **This demo doesn't perform human/hand detection**. Our model requires a centered target in the image. 
 - As **METRO is a data-driven approach**, it may not perform well if the test samples are very different from the training data. 
 - **METRO is a mesh-specific approach**. For example, our hand model is trained only on right-hand data, and therefore it doesn't perform well on the left-hand images. How to develop a unified model for different 3D objects is an interesting future work. 



