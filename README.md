# MoireDet
cuda:10.0;
pytorch 1.4;
torchvision 0.5;

## Modern checkpoint inference

本项目的复现分支提供了不依赖作者机器绝对路径的现代单图/视频入口。它们使用
checkpoint 内置的完整模型配置，默认按训练数据的 RGB 顺序预处理；仓库中的
`PSENet_100_loss0.000000.pth` 是社区备份权重，并非作者官方重新发布的权重。

单图：

```bash
cd MoireDet
conda run --no-capture-output -n pytorch python script/infer.py \
  --input script/00002423.png \
  --checkpoint script/PSENet_100_loss0.000000.pth \
  --output /path/to/prediction.png \
  --comparison-output /path/to/comparison.png \
  --device cuda:0 --color-order rgb
```

视频 demo 会逐帧推理，并输出“左侧原视频、右侧灰度摩尔纹边缘预测图”的 MP4；
同时默认保存同名 JSON，记录帧数、FPS、设备和耗时。它不保留音频，也不产生论文
意义上的真值标注：

```bash
conda run --no-capture-output -n pytorch python script/infer_video.py \
  --input /path/to/moire_video.mp4 \
  --checkpoint script/PSENet_100_loss0.000000.pth \
  --output /path/to/moire_video_demo.mp4 \
  --device cuda:0 --color-order rgb --display-height 720
```

如果手机视频方向或编码导致 OpenCV 无法读取，可先转为常见的 MP4/H.264 或使用
`--codec avc1` 尝试写出；视频演示的缩放、拼接和编码属于展示工程，不改变论文模型。


# MoireScape*

Our proposed MoireScape dataset contains two subsets:
 - MoireScape-real: It contains 500 real image pairs for evaluating moiré edge map estimation. Each pair includes a real camera-captured screen image and its moiré layer with the setup in Fig. 3. To extract moire edge map from a moire layer, the more_layer_segmentation tool can be used.
 - MoireScape-synthetic: It contains 18,147 different moiré layers and 4,000 natural images. After varying moiré layers and the their combinations, 50,000 synthetic triplets are collected for the purpose of training (90%) and testing(10%). Each triplet contains a natural image, a moiré layer, and their synthetic mixture.

*: Since each subset surpass 25M, please download them via the online disc link in each text file. 
 
## Citation

If you benefit from this work, please cite the mentioned and our paper:

	@article{Yang2023Moire,
		author = {Cong Yang and Zhenyu Yang and Yan Ke and Tao Chen and Marcin Grzegorzek and John See},
		title = {Doing More With Moiré Pattern Detection in Digital Photos},
		journal = {IEEE Transactions on Image Processing},
            volume = {32},
            pages = {694-708},
            year = {2023}
	}
