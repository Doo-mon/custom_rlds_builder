import os
import random
import cv2
import yaml
import torch
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image
from torchvision.transforms import ToTensor
from typing import Iterator, Tuple, Any, Optional


'''
将 ARIO 格式的数据转变为 RLDS 格式

为 seawave_real 数据集添加深度图 并重新生成 rlds 数据集
'''

class SeawaveRealDataset(tfds.core.GeneratorBasedBuilder):
    
    VERSION = tfds.core.Version('1.0.0')
    # RELEASE_NOTES = {'1.0.0': 'Initial release.',}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ori_dataset_name = "collection-real-SeaWave"
        self.ARIO_series_name = "series-1"
        self.ori_data_dir = f"/data1/zhanzhihao/openvla_data/{self.ori_dataset_name}/{self.ARIO_series_name}/"
        
        self.episode_count = 0 # 考虑到有些数据集并没有 episode_id

        self.has_val = True

        self.task_list = [name for name in os.listdir(self.ori_data_dir) if os.path.isdir(os.path.join(self.ori_data_dir, name))]
        self.task_list.sort()

        if self.has_val:
            self.train_dict, self.val_dict = self._split_train_val(is_split = True, is_random = True, trian_split_ratio = 0.9)
        else:
            self.train_dict = self._split_train_val(is_split = False)
         
        self.device_0 = "cuda:0"
        self.device_1 = "cuda:1"

        # 加载深度模型
        self.depth_model = torch.hub.load("/data1/zhanzhihao/ZoeDepth/", "ZoeD_N", source="local", pretrained=True)
        self.depth_model.eval()
        self.depth_model = self.depth_model.to(self.device_1)


    def _split_train_val(self, is_split = False, is_random = True, trian_split_ratio = 0.9)-> Tuple[dict, Optional[dict]]:
        
        task_dict = {}
        for t_name in self.task_list:
            task_path = self.ori_data_dir + t_name + "/"
            yaml_file = task_path + "description.yaml"
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)  # 加载 YAML 文件
                des = data["instruction_EN"]
            epi_list = [(task_path + name + "/") for name in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, name))]
            task_dict[des] = epi_list

        if not is_split:
            return task_dict
        
        keys = list(task_dict.keys())
        if is_random:
            random.shuffle(keys)
        split_index = int(len(keys) * trian_split_ratio)

        train_dict = {key: task_dict[key] for key in keys[:split_index]}
        val_dict = {key: task_dict[key] for key in keys[split_index:]}
        return train_dict, val_dict
      
        
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
                            shape=(8,),  
                            dtype=np.float32,
                            doc='dataa robot has 7x + gripper',
                        )
                    }),
                    # 'action': tfds.features.Tensor(
                    #     shape=(7,),
                    #     dtype=np.float32,
                    #     doc='Robot action, consists of [7x joint velocities, '
                    #         '2x gripper velocities, 1x terminate episode].',
                    # ),
                    # 'discount': tfds.features.Scalar(
                    #     dtype=np.float32,
                    #     doc='Discount if provided, default to 1.'
                    # ),
                    # 'reward': tfds.features.Scalar(
                    #     dtype=np.float32,
                    #     doc='Reward if provided, 1 on final step for demos.'
                    # ),
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
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='copied from origin bridge dataset'
                    # ),
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
        def _iter_data(data_dict):
            for k, v_list in data_dict.items():
                for v in v_list:
                    yield k, v
                    
        def _extract_frames(video_path):
            frames = []
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return frames
            
        def _extract_state_list(episode):
            arm_joint = []
            for i in range(7):
                e_list = []
                with open(episode + f"right_master_arm_joint-{i}.txt",'r') as f:
                    for l in f:
                        fst , sed = l.strip().split(" ")
                        e_list.append((int(fst), sed))
                e_list.sort(key=lambda x: x[0])
                arm_joint.append([second for _, second in e_list])

            e_list = []
            with open(episode + "right_master_gripper.txt",'r') as f:
                for l in f:
                    fst , sed = l.strip().split(" ")
                    e_list.append((int(fst), sed))
                e_list.sort(key=lambda x: x[0])
            arm_joint.append([second for _, second in e_list])

            arm_joint = [list(x) for x in zip(*arm_joint)] # transposed list
            return arm_joint

        def _process_origin_image(ori_img):
            img = cv2.resize(ori_img, (256, 256))
            return img

        def _generate_depth_image(ori_img):
            ori_torch = ToTensor()(Image.fromarray(ori_img)).unsqueeze(0).to(self.device_1)
            depth_torch = self.depth_model.infer(ori_torch) # 1 1 256 256
            depth_np = depth_torch.squeeze().detach().cpu().numpy() # 256 256 float32
            depth_tf = tf.convert_to_tensor(depth_np) # 256 256
            return tf.expand_dims(depth_tf, axis=-1) # 256 256 1

        def _parse_example(des, episode):
            video_dir = episode + "cam-1/rgb.mp4"
            # timestamp_dir = episode + "cam-1/timestamps.npy"
            # result_dir = episode + "result.txt"
            # gripper_dir = episode + "right_master_gripper.txt"
            # 提取状态列表
            arm_joint_state_list = _extract_state_list(episode)
            # 提取视频
            steps = _extract_frames(video_dir) # np 480 640 3
            steps_len = len(steps)
            new_episode = []
            self.episode_count += 1
            for i, img in enumerate(steps):
                image_0 = _process_origin_image(img)
                depth_0 = _generate_depth_image(image_0)
                new_episode.append({
                    'observation': {
                        'image_0': image_0,
                        'depth_0': np.array(depth_0),
                        'state':np.array(arm_joint_state_list[i], dtype = np.float32),
                    },
                    'is_first':np.array(True if i==0 else False),
                    'is_last': np.array(True if i==steps_len else False),
                    'is_terminal': np.array(True if i==steps_len else False),
                    'language_instruction': np.array(tf.constant(des)),
                })
            
            sample = {
                'steps': new_episode,
                'episode_metadata': {
                    'file_path': episode,
                    'has_image_0':np.array(True, dtype = bool),
                    'has_depth_0':np.array(True, dtype = bool),
                    'has_language': np.array(True, dtype = bool),
                    'episode_id': np.array(self.episode_count, dtype = np.int32)
                }
            }

            return f"episode_{self.episode_count}", sample

       
        # 根据传进去的不同 episode 执行对应的函数获得不同的返回值
        data_dict = self.train_dict if (split_mode=="train") else self.val_dict

        for des, episode in _iter_data(data_dict):
            # print(des, episode)
            yield _parse_example(des, episode)

        # 数据集很大的话考虑使用 beam 进行并行数据处理(this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
