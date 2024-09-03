# V10Drone-YOLOv10

this is source code of V10Drone.
It includes YOLOv10, KAN, 4heads small object detector

## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate visdrone
pip install -r requirements.txt
pip install -e .
```


## Validation

```
yolo val model=<your weight> data=VisDrone.yaml batch=256 half=True imgsz=640
```


## Training 
```
yolo detect train data= ~/ultralytics/ultralytics/cfg/datasets/visdrone.yaml model= ~/yolov10/ultralytics/cfg/models/kan/konv_detr_dwconv_yolov10.yaml epochs=350 imgsz=640 workers=36 batch=30 device=0,1,2 name= konv_detr_dwconv_yolov10
```




## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics), [RT-DETR](https://github.com/lyuwenyu/RT-DETR) and YOLOv10.

Thanks for the great implementations! 


# 
# 
