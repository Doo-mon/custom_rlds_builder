# RLDS Dataset Conversion

此 repo 展示了将多种格式的数据集 转换为 RLDS 格式，以便集成 X-embodiment 实验。

（此外还可以支持生成不同的数据插入到原数据中，例如为数据集生成深度图、点云等）

目前提供以下示例，可根据需要对代码进一步修改： 

* 自定义数据集
*  rlds 格式 （Open-X : https://robotics-transformer-x.github.io/ ）
*  ario 格式 （ARIO : https://imaei.github.io/project_pages/ario/ ）


## 安装

**对于 rlds_env 环境 可以使用提供的文件进行简单安装 `pip install -r requirement.txt`**

也可以参考原 repo 的安装 https://github.com/kpertsch/rlds_dataset_builder

**24-10-27** 最新的代码使用 Depth-Anything-V2 进行深度图生成（参考 https://github.com/DepthAnything/Depth-Anything-V2 ）
* 目前使用 large 模型（335.3M），giant 模型（1.3B）暂未公开
* 直接 `clone` 仓库，在代码中填入对应的 **仓库路径** 和 **模型ckpt** 即可使用
* **优点：更加方便直接，并且无需额外配置环境，可在rlds_env的环境下使用**

旧示例代码使用的深度图生成模型的使用 （参考 https://github.com/isl-org/ZoeDepth ）
* 该模型由于也调用了外部下载，需要进行一点手动的修改

**可根据自己需要更换深度估计模型**

## 测试

测试初始环境是否能正常使用 可以使用以下命令
```shell
cd example_data
python3 create_example_data.py
tfds build
```


## 使用

切换到具体的目录，执行以下命令即可

```shell
export CUDA_VISIBLE_DEVICES=1,2
conda activate rlds_env

cd /data1/zhanzhihao/custom_rlds_builder/my_robotdata

# --data_dir 为生成的数据集保存路径
tfds build --data_dir "/data1/zhanzhihao/openvla_data"

# 后台执行命令
nohup tfds build --data_dir "/data1/zhanzhihao/openvla_data" > ./output.log 2>&1 &
```

补充几点重要的：

1. 名字要对上，包括比如 seawave_real 文件夹 就对应 seawave_real_dataset_builder.py
2. 每个数据集类里面设置的meta信息也要和最后generate的一致
3. _generate_examples() 函数返回的第一个参数一定要不一致（否则将在最后一步save的时候出问题）

## 并行数据处理

默认情况下，数据集转换是单线程的。如果需要解析大型数据集，则可以使用并行处理。

为此，请将 `_generate_examples()` 最后两行替换为注释掉的 `beam` 命令。这将使用 **Apache Beam** 并行化数据处理。

在开始处理之前，需要在 `setup.py` 填写相应的数据集的名称并安装数据集包 `pip install -e .`

可以利用 `direct_num_workers` 指定处理的数量

```shell
tfds build --data_dir "/data1/zhanzhihao/openvla_data" --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=2"
```

## 其他

**国内服务器难以进行外网下载的问题**

（注意数据集版本不同，位置也会有不同！！！）

**tf在构建数据集的时候会首先搜索一下云上有没有相应的数据集，如果服务器访问不了外网的话，这一步将会等待非常久，所以需要稍微改动一下 tensorflow_datasets==4.9.2 的源码**:
1. 修改一：将 `tensorflow_datasets/core/dataset_builder.py`  里面 275行 `self.info.initialize_from_bucket()` 注释掉 或者 替换成 `pass`
2. 修改二：将 `tensorflow_datasets/scripts/cli/build.py`  里面的 590行 `tfds.download.DownloadConfig` 类的构建传入参数 `try_download_gcs` 设为 `False`
