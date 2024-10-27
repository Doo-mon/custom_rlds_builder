# RLDS Dataset Conversion

此 repo 展示了将多种格式的数据集 转换为 RLDS 格式，以便集成 X-embodiment 实验。

（此外还可以支持生成不同的数据插入到原数据中，例如为数据集生成深度图、点云等）

目前提供以下示例，可根据需要对代码进一步修改： 

* 自定义数据集
*  rlds 格式 （Open-X : https://robotics-transformer-x.github.io/ ）
*  ario 格式 （ARIO : https://imaei.github.io/project_pages/ario/ ）


## 安装

rlds_env 环境参考原 repo 的安装 https://github.com/kpertsch/rlds_dataset_builder


**24-10-27** 最新的代码使用 Depth-Anything-V2 进行深度图生成（参考 https://github.com/DepthAnything/Depth-Anything-V2 ）
* 目前使用 large 模型（335.3M），giant 模型（1.3B）暂未公开
* 直接 `clone` 仓库，在代码中填入对应的 **仓库路径** 和 **模型ckpt** 即可使用
* **优点：更加方便直接，并且无需额外配置环境，可在rlds_env的环境下使用**

旧示例代码使用的深度图生成模型的使用 （参考 https://github.com/isl-org/ZoeDepth ）
* 该模型由于也调用了外部下载，需要进行一点手动的修改

**可根据自己需要更换深度估计模型**



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



## 其他

**国内服务器难以进行外网下载的问题**

**tf在构建数据集的时候会首先搜索一下云上有没有相应的数据集，如果服务器访问不了外网的话，这一步将会等待非常久，所以需要稍微改动一下 tensorflow_datasets 的源码**:
1. 修改一：将 tensorflow_datasets/core/dataset_builder.py  里面275行的函数注释掉（可替换成 pass）
2. 修改二：将 tensorflow_datasets/scripts/cli/build.py  里面的590行 DownloadConfig 类的参数 try_download_gcs 设为 False




<!-- ## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.


## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Converting your Own Dataset to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Rename Dataset**: Change the name of the dataset folder from `example_dataset` to the name of your dataset (e.g. robo_net_v2), 
also change the name of `example_dataset_dataset_builder.py` by replacing `example_dataset` with your dataset's name (e.g. robo_net_v2_dataset_builder.py)
and change the class name `ExampleDataset` in the same file to match your dataset's name, using camel case instead of underlines (e.g. RoboNetV2).

2. **Modify Features**: Modify the data fields you plan to store in the dataset. You can find them in the `_info()` method
of the `ExampleDataset` class. Please add **all** data fields your raw data contains, i.e. please add additional features for 
additional cameras, audio, tactile features etc. If your type of feature is not demonstrated in the example (e.g. audio),
you can find a list of all supported feature types [here](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?hl=en#classes).
You can store step-wise info like camera images, actions etc in `'steps'` and episode-wise info like `collector_id` in `episode_metadata`.
Please don't remove any of the existing features in the example (except for `wrist_image` and `state`), since they are required for RLDS compliance.
Please add detailed documentation what each feature consists of (e.g. what are the dimensions of the action space etc.).
Note that we store `language_instruction` in every step even though it is episode-wide information for easier downstream usage (if your dataset
does not define language instructions, you can fill in a dummy string like `pick up something`).

3. **Modify Dataset Splits**: The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.).
If your dataset defines a train vs validation split, please provide the corresponding information to `_generate_examples()`, e.g. 
by pointing to the corresponding folders (like in the example) or file IDs etc. If your dataset does not define splits,
remove the `val` split and only include the `train` split. You can then remove all arguments to `_generate_examples()`.

4. **Modify Dataset Conversion Code**: Next, modify the function `_generate_examples()`. Here, your own raw data should be 
loaded, filled into the episode steps and then yielded as a packaged example. Note that the value of the first return argument,
`episode_path` in the example, is only used as a sample ID in the dataset and can be set to any value that is connected to the 
particular stored episode, or any other random value. Just ensure to avoid using the same ID twice.

5. **Provide Dataset Description**: Next, add a bibtex citation for your dataset in `CITATIONS.bib` and add a short description
of your dataset in `README.md` inside the dataset folder. You can also provide a link to the dataset website and please add a
few example trajectory images from the dataset for visualization.

6. **Add Appropriate License**: Please add an appropriate license to the repository. 
Most common is the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license -- 
you can copy it from [here](https://github.com/teamdigitale/licenses/blob/master/CC-BY-4.0).

That's it! You're all set to run dataset conversion. Inside the dataset directory, run:
```
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.


### Parallelizing Data Processing
By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use 
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package 
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:
```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```
You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py <name_of_your_dataset>
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 

## Add Transform for Target Spec

For X-embodiment training we are using specific inputs / outputs for the model: input is a single RGB camera, output
is an 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination
action.

The final step in adding your dataset to the training mix is to provide a transform function, that transforms a step
from your original dataset above to the required training spec. Please follow the two simple steps below:

1. **Modify Step Transform**: Modify the function `transform_step()` in `example_transform/transform.py`. The function 
takes in a step from your dataset above and is supposed to map it to the desired output spec. The file contains a detailed
description of the desired output spec.

2. **Test Transform**: We provide a script to verify that the resulting __transformed__ dataset outputs match the desired
output spec. Please run the following command: `python3 test_dataset_transform.py <name_of_your_dataset>`

If the test passes successfully, you are ready to upload your dataset!

## Upload Your Data

We provide a Google Cloud bucket that you can upload your data to. First, install `gsutil`, the Google cloud command 
line tool. You can follow the installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

Next, authenticate your Google account with:
```
gcloud auth login
``` 
This will open a browser window that allows you to log into your Google account (if you're on a headless server, 
you can add the `--no-launch-browser` flag). Ideally, use the email address that
you used to communicate with Karl, since he will automatically grant permission to the bucket for this email address. 
If you want to upload data with a different email address / google account, please shoot Karl a quick email to ask 
to grant permissions to that Google account!

After logging in with a Google account that has access permissions, you can upload your data with the following 
command:
```
gsutil -m cp -r ~/tensorflow_datasets/<name_of_your_dataset> gs://xembodiment_data
``` 
This will upload all data using multiple threads. If your internet connection gets interrupted anytime during the upload
you can just rerun the command and it will resume the upload where it was interrupted. You can verify that the upload
was successful by inspecting the bucket [here](https://console.cloud.google.com/storage/browser/xembodiment_data).

The last step is to commit all changes to this repo and send Karl the link to the repo.

**Thanks a lot for contributing your data! :)** -->
