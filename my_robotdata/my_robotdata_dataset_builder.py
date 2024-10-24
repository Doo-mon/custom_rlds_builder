import os
import glob
import torch
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib

from PIL import Image
from torchvision.transforms import ToTensor
from typing import Iterator, Tuple, Any


'''
为原来的 bridge 数据集添加深度图 并重新生成 rlds 数据集

'''

class MyRobotDataset(tfds.core.GeneratorBasedBuilder):
    
    VERSION = tfds.core.Version('1.0.0')
    # RELEASE_NOTES = {'1.0.0': 'Initial release.',}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ori_dataset_name = "bridge_orig_for_test" # bridge_dataset
        self.ori_dataset_version = "1.0.0"   # "1.0.0"
        
        self.ori_data_dir = f"/data1/zhanzhihao/openvla_data/{self.ori_dataset_name}/{self.ori_dataset_version}"
        self.origin_data_builder = tfds.builder_from_directory(builder_dir = self.ori_data_dir)

        self.has_val = True 
        self.episode_count = 0

        # for gpus 这里用两个GPU 单个GPU在加载深度模型的时候会爆显存
        self.device_0 = "cuda:0"
        self.device_1 = "cuda:1"

        # 加载深度模型
        self.depth_model = torch.hub.load("/data1/zhanzhihao/ZoeDepth/", "ZoeD_N", source="local", pretrained=True)
        self.depth_model.eval()
        self.depth_model = self.depth_model.to(self.device_1)
        
        
    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'depth_0': tfds.features.Tensor(
                            shape=(256, 256, 1), 
                            dtype=np.float32,
                            doc='depth for Main camera.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),  
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='copied from origin bridge dataset'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc="Path to the original data file."
                    ),
                    'has_image_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True if image0 exists in observation, otherwise dummy value."
                    ),
                    'has_depth_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True if depth0 exists in observation, otherwise dummy value."
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True if language exists in observation, otherwise empty string."
                    ),
                    "episode_id":tfds.features.Scalar(
                        dtype=np.int32,
                        doc="ID of episode in file_path."
                    )
                }),
            }))


    def _split_generators(self, dl_manager: tfds.download.DownloadManager):

        if self.has_val :
            return {
                'train': self._generate_examples(split_mode="train"),
                'val': self._generate_examples(split_mode="val"),
            }
        else:
            return {
                'train': self._generate_examples(split_mode="train"),
            }


    def _generate_examples(self, split_mode="train") -> Iterator[Tuple[str, Any]]:
        def _generate_depth_image(ori_img):
            '''
                因为深度模型是torch的 需要转变为torch张量再输入 然后保存的格式是tf张量 所以这里显得很繁琐....
                tf张量和torch张量之间的转换需要以np为中介...
            '''
            ori_torch = ToTensor()(Image.fromarray(np.array(ori_img))).unsqueeze(0).to(self.device_1)
            depth_torch = self.depth_model.infer(ori_torch) # 1 1 256 256
            depth_np = depth_torch.squeeze().detach().cpu().numpy() # 256 256 float32
            depth_tf = tf.convert_to_tensor(depth_np) # 256 256
            return tf.expand_dims(depth_tf, axis=-1) # 256 256 1


        def _parse_example(episode):
            steps = episode["steps"]
            new_episode = [] # 存放新episode的step列表
            self.episode_count += 1
            
            for i, step in enumerate(steps):
                image_0 = step['observation']['image_0'] # tf.tensor 256 256 3
                depth_0 = _generate_depth_image(image_0) # tf.tensor 256 256 1
                new_episode.append({
                    'observation': {
                        'image_0': np.array(image_0),
                        'depth_0': np.array(depth_0),
                        'state': np.array(step['observation']['state']),
                    },
                    'action': np.array(step['action']),
                    'discount': np.array(step['discount']),
                    'reward': np.array(step['reward']),
                    'is_first':np.array(step['is_first']),
                    'is_last': np.array(step['is_last']),
                    'is_terminal': np.array(step['is_terminal']),
                    'language_instruction': np.array(step['language_instruction']),
                    'language_embedding': np.array(step['language_embedding']),
                })
            
            sample = {
                'steps': new_episode,
                'episode_metadata': {
                    'file_path': self.ori_data_dir,
                    'has_image_0':np.array(True, dtype = bool),
                    'has_depth_0':np.array(True, dtype = bool),
                    'has_language': np.array(True, dtype = bool),
                    'episode_id': np.array(episode["episode_metadata"]['episode_id'])
                }
            }

            return f"episode_{self.episode_count}", sample

        # 这条语句只会执行一次
        self.origin_data_dataset = self.origin_data_builder.as_dataset(split = split_mode)

        # 根据传进去的不同 episode 执行对应的函数获得不同的返回值
        for episode in self.origin_data_dataset:
            yield _parse_example(episode)

        # 数据集很大的话考虑使用 beam 进行并行数据处理(this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

