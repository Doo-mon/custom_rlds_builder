import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tqdm
import wandb
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

"""
用于可视化自定义生成的数据集

"""

WANDB_ENTITY = "doo_mon"
WANDB_PROJECT = 'vis_rlds'


sample_num = 30
vis_num = 6
downsample_n = 10

dataset_name = 'seawave_real_dataset' # seawave_real_dataset    bridge_orig_for_test
dataset_version = "1.0.0"
data_dir = f'/data1/zhanzhihao/openvla_data/{dataset_name}/{dataset_version}/'

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY, 
               project=WANDB_PROJECT,
               name=f"{dataset_name} -- vis",
               )
else:
    render_wandb = False


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[...]
    img[invalid_mask] = background_color
    img = img[:, :, :-1]

    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img



print(f"Visualizing data from dataset: {dataset_name}")
builder = tfds.builder_from_directory(builder_dir = data_dir)
ds = builder.as_dataset(split=f'train[:{sample_num}]').shuffle(sample_num)


# 可视化图片和深度图
for i, episode in enumerate(ds.take(vis_num)):
    images = [step['observation']["image_0"].numpy() for step in episode['steps']]
    depths = [colorize(step['observation']["depth_0"].numpy()) for step in episode['steps']]
    image_strip = np.concatenate(images[::downsample_n], axis=1)
    depth_strip = np.concatenate(depths[::downsample_n], axis=1)
    concat_i_d = np.vstack((image_strip, depth_strip))

    first_step = next(iter(episode["steps"]))
    caption = first_step['language_instruction'].numpy().decode() + f' (downsampled {downsample_n}x)'
    if render_wandb:
        wandb.log({f'episode_{i}': wandb.Image(concat_i_d, caption=caption)})
    else:
        print("Can not render in wandb")
        break
    

print("finished....")