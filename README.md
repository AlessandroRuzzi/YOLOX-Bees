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

**Step 1:** Download a yolox pre-trained checkpoint from the table below and put it in the folder ``YOLOX-Bees/checkpoints``.

#### Standard Models.

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |

**Step 2:** Based on the checkpoint you downloaded you will choose a different experiment file. They are located in ``/YOLOX-Bees/exps/default/`` and you can choose between ``yolox_s``, ``yolox_m``, ``yolox_l`` and ``yolox_x``.

**Step 3:** Download from Azure the folder ``/TODO:path to add/bees_all`` and put it inside the folder ``YOLOX_Bees/datasets/``.

**Step 4:** Run the following command to train yolox using a single GPU(it can only be trained with GPUs)
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

**Step 1:** Download a yolox checkpoint from Azure (TODO: add path to checkpoints) or use one checkpoint that you produced and put it in the folder ``YOLOX-Bees/checkpoints``. 

**Step 2:** Then download the evaluation datasets from Azure (TODO: add path to checkpoints) and put them in the folder ``YOLOX-Bees/datasets/``.

**Step 3:** Open the file ``YOLOX-Bees/exps/default/yolox_bees_eval.py`` and modify ``self.depth`` and ``self.width`` based on the checkpoint you have downloaded ( yolox_x -> [1.33, 1.25] , yolox_l -> [1.0, 1.0] , yolox_m -> [0.67, 0.75] , yolo_s -> [0.33, 0.50]] )

**Step 4:** Run the following command to obtain predictions for all the datasets
```console
python evaluation.py image -f exps/default/yolox_bees_eval.py -c checkpoints/YOUR_CHECKPOINT.pth --tsize 640 --save_result
```
**Step 5:**
At the end you will find a file called ``mAP_results.txt`` together with an output file for each dataset in the folder ``YOLOX-Bees/map/output/``, while you will find images with bounding boxes predicted by the model in the folder ``YOLOX-Bees/YOLOX_outputs/yolox_bees_eval/``.
</details>


# How to evaluate a model different from yolox
<details>
<summary>Create the detection-results files</summary>

**Step 1:** Use your model to create a separate detection-results text file for each image for each dataset.

**Step 2:** Use **matching names** for the files (e.g. image: "image_1.jpg", detection-results: "image_1.txt").

**Step 3:** In these files, each line should be in the following format:
```
    <class_name> <confidence> <left> <top> <right> <bottom>
```

**Step 4:** E.g. "image_1.txt":
```
    tvmonitor 0.471781 0 13 174 244
    cup 0.414941 274 226 301 265
    book 0.460851 429 219 528 247
    chair 0.292345 0 199 88 436
    book 0.269833 433 260 506 336
```
**Step 5:** Put all the files in the folder ``YOLOX-Bees/map/input/DATASET_NAME/detection-results``, where ``DATASET_NAME`` can be for example ``Chueried_Hive01``.

To know all the datasets name you can refer to lines 30 - 41 of the file ``evaluation.py``.

At the end the folder ``YOLOX-Bees/map/input/`` should have the following structure:
```
input
   |——————Chueried_Hive01
   |        └——————detection-results
   |        
   |——————ClemensRed
   |        └——————detection-results
   | 
   |——————Doettingen_Hive1
   |        └——————detection-results
   | 
   |——————Echolinde
   |        └——————detection-results
   | 
   |——————Erlen_diago
   |        └——————detection-results
   | 
   |——————Erlen_front
   |        └——————detection-results
   | 
   |——————Erlen_Hive11
   |        └——————detection-results
   | 
   |——————Erlen_smart
   |        └——————detection-results
   | 
   |——————Froh14
   |        └——————detection-results
   | 
   |——————Froh23
   |        └——————detection-results
   | 
   |——————Hempbox
   |        └——————detection-results
   | 
   |——————UnitedQueens
   |        └——————detection-results
```
</details>
<details>
<summary>Evaluation</summary>

**Step 1:** Download the evaluation datasets from Azure (TODO: add path to checkpoints) and put them in the folder ``YOLOX-Bees/datasets/`` (we need it to create ground truth labels).

**Step 2:** Run the following command to obtain predictions for all the datasets
```console
python evaluation_no_yolox.py image -f exps/default/yolox_bees_eval.py  --tsize 640
```

**Step 3:**
At the end you will find a file called ``mAP_results.txt`` together with an output file for each dataset in the folder ``YOLOX-Bees/map/output/``.

</details>

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
