<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

# How to reproduce results


## Quick Start

<details>
<summary>Installation</summary>

**Step 1:** Install YOLOX-Bees.
```console
git clone https://github.com/AlessandroRuzzi/YOLOX-Bees
cd YOLOX-Bees
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```

**Step 2:** Install [pycocotools](https://github.com/cocodataset/cocoapi).

```console
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

</details>

<details>
<summary>Training</summary>

**Step 1:** Download a yolox checkpoint from Azure (TODO: add path to checkpoints) and put it in the folder ``YOLOX-Bees/checkpoints``. 

**Step 2:** Based on the checkpoint you downloaded you will choose a different experiment file -> they are located in ``/YOLOX-Bees/exps/default/``.

**Step 3:** Download from Azure the folder /path to add/bees_all and put it inside the folder ``YOLOX_Bees/datasets/``

**Step 4:** Run the following commando to train yolox using a single GPU(it can only be trained with GPUs)
```console
python tools/train.py -f exps/default/YOUR_EXP_FILE.py -d 1 -b 4 --fp16 -o -c checkpoints/YOUR_CHECKPOINT.pth
```
If you are using Euler cluster you can run:

```console
bsub -W 24:00 -o log_test -R "rusage[mem=32000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python tools/train.py -f exps/default/YOUR_EXP_FILE.py -d 1 -b 4 --fp16 -o -c checkpoints/YOUR_CHECKPOINT.pth
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8
* --fp16: mixed precision training 

**Step 5:** Once the train ends you will find in the folder ``/YOLOX-Bees/YOLOX_outputs/YOUR_EXP_NAME/`` the best checkpoint (evaluated on the validation set) and the last epoch checkpoint.

</details>


<details>
<summary>Evaluation</summary>

**Step 1:**

TODO

</details>


# How to evaluate a model different from yolox

TODO

# Deployment


1.  [MegEngine in C++ and Python](./demo/MegEngine)
2.  [ONNX export and an ONNXRuntime](./demo/ONNXRuntime)
3.  [TensorRT in C++ and Python](./demo/TensorRT)
4.  [ncnn in C++ and Java](./demo/ncnn)
5.  [OpenVINO in C++ and Python](./demo/OpenVINO)


# Third-party resources
* The ncnn android app with video support: [ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
* YOLOX with Tengine support: [Tengine](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolox.cpp) from [BUG1989](https://github.com/BUG1989)
* YOLOX + ROS2 Foxy: [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) from [Ar-Ray](https://github.com/Ar-Ray-code)
* YOLOX Deploy DeepStream: [YOLOX-deepstream](https://github.com/nanmi/YOLOX-deepstream) from [nanmi](https://github.com/nanmi)
* YOLOX MNN/TNN/ONNXRuntime: [YOLOX-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolox.cpp)、[YOLOX-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolox.cpp) and [YOLOX-ONNXRuntime C++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolox.cpp) from [DefTruth](https://github.com/DefTruth)
* Converting darknet or yolov5 datasets to COCO format for YOLOX: [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO) from [Daniel](https://github.com/znsoftm)

# Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
