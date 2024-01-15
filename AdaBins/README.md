# AdaBins
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adabins-depth-estimation-using-adaptive-bins/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=adabins-depth-estimation-using-adaptive-bins) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adabins-depth-estimation-using-adaptive-bins/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=adabins-depth-estimation-using-adaptive-bins)

Official implementation of [Adabins: Depth Estimation using adaptive bins](https://arxiv.org/abs/2011.14141)

[github repo](https://github.com/shariqfarooq123/AdaBins)
## Download links
* You can download the pretrained models "AdaBins_nyu.pt" and "AdaBins_kitti.pt" from [here](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing)
* You can download the predicted depths in 16-bit format for NYU-Depth-v2 official test set and KITTI Eigen split test set [here](https://drive.google.com/drive/folders/1b3nfm8lqrvUjtYGmsqA5gptNQ8vPlzzS?usp=sharing)

## Colab demo 

<p>
<a href="https://colab.research.google.com/drive/1oxHflMh6eAJS7BhvP1amHvuBSirlS5Vl?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

## dataset
```Python
    #dataset path training
    parser.add_argument("--data_path", default='./dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='./dataset/nyu/sync/', type=str,
                        help="path to dataset")
    #dataset test
    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')   
    #dataset path eval
    parser.add_argument('--data_path_eval',
                        default="./dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="./dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
```

## how to train
关于训练，主要就是修改一下数据集的位置。
其次是加载模型，由于模型是从github上加载，然而不管是否使用代理都会出现加载问题，所以将其使用的仓库下载下来然后本地加载即可
```python
    #url
    # basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
    #找到仓库地址，下载在 repo_or_dir ="/home/xxx/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master"
    #加载
    basemodel = torch.hub.load(repo_or_dir, basemodel_name, source='local',pretrained=True)
```
使用GPU训练，对显存要求很高，这里的6G满足不了

## Inference
Move the downloaded weights to a directory of your choice (we will use "./pretrained/" here). You can then use the pretrained models like so:

```python
from models import UnetAdaptiveBins
import model_io
from PIL import Image

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

# NYU
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
pretrained_path = "./pretrained/AdaBins_nyu.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)

# KITTI
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "./pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)
```
Note that the model returns bin-edges (instead of bin-centers).

**Recommended way:** `InferenceHelper` class in `infer.py` provides an easy interface for inference and handles various types of inputs (with any prepocessing required). It uses Test-Time-Augmentation (H-Flips) and also calculates bin-centers for you:
```python
from infer import InferenceHelper

infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a batched rgb tensor
example_rgb_batch = ...  
bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/path/to/input/dir/containing_only_images/", "path/to/output/dir/")

```
## TODO:
* Add instructions for Evaluation and Training.
* Add UI demo
* Remove unnecessary dependencies
