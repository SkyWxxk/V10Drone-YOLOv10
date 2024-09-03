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

## Result
| Method   | Precision (%) | Recall (%) | mAP50 (%) | mAP (%) | Parameters (M) | FLOPs (G) |
|----------|---------------|------------|-----------|---------|----------------|-----------|
| YOLOv5s  | 51.5          | 37.5       | 39.3      | 23.5    | 9.1            | 24.1      |
| YOLOv6s  | 50.0          | 36.7       | 37.6      | 22.5    | 17.2           | 44.2      |
| YOLOv8s  | 50.4          | 37.8       | 38.4      | 22.8    | 11.2           | 28.6      |
| YOLOv10s | 50.1          | 37.5       | 38.5      | 23.0    | 8.0            | 24.8      |
| YOLOv5m  | 53.5          | 40.3       | 42.2      | 25.7    | 25.0           | 64.4      |
| YOLOv6m  | 53.6          | 39.1       | 41.1      | 25.1    | 52.0           | 161.6     |
| YOLOv8m  | **55.0**      | 40.5       | 42.3      | 25.8    | 25.8           | 79.1      |
| YOLOv10m | 54.0          | 40.6       | 42.2      | 25.7    | 16.4           | 63.5      |
| V10Drone | 53.1          | **41.7**   | **43.4**  | **26.3**| **7.9**        | 48.9      |



## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics), [RT-DETR](https://github.com/lyuwenyu/RT-DETR) and YOLOv10.

Thanks for the great implementations! 


# 
# 
