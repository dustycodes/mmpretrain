from mmpretrain.apis import init_model, inference_model
from mmcv.transforms import Compose, Resize
import torch
import glob
import os
import mmcv
from mmpretrain import inference_model
from mmpretrain import ImageClassificationInferencer


cfg = 'configs/resnet/resnet50_8xb256-rsb-a3-100e_transformers.py'
weight = 'work_dirs/resnet50_8xb256-rsb-a3-100e_transformers/epoch_297.pth'

inferencer = ImageClassificationInferencer(model=cfg, pretrained=weight, device='cuda')

# # Specify the path to model config and checkpoint file
# config_file = 'configs/resnet/resnet50_8xb256-rsb-a3-100e_transformers.py'
# checkpoint_file = 'work_dirs/resnet50_8xb256-rsb-a3-100e_transformers/epoch_297.pth'

# # build the model from a config file and a checkpoint file
# model = init_model(config_file, checkpoint_file, device='cuda:0')

# List of images
img_list = glob.glob(os.path.join("data/infer-transformers/test", "*"))

results = inferencer(img_list)

i = 0
for result, img in zip(results, img_list):
    # if result["pred_label"] == 0 and result["pred_score"] > 0.7:
    #     continue
    if result["pred_label"] == 0:
        continue
    print(f"Image: {img} Result: {result}")
    i+=1
    
print(f"{i}/{len(img_list)}")
