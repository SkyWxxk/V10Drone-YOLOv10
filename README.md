# V10Drone-YOLOv10

this is source code of V10Drone.
It concludes YOLOv10, KAN, 4heads small object detector

## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```
## Demo
```
python app.py
# Please visit http://127.0.0.1:7860
```

## Validation
[`yolov10n`](https://huggingface.co/jameslahm/yolov10n)  [`yolov10s`](https://huggingface.co/jameslahm/yolov10s)  [`yolov10m`](https://huggingface.co/jameslahm/yolov10m)  [`yolov10b`](https://huggingface.co/jameslahm/yolov10b)  [`yolov10l`](https://huggingface.co/jameslahm/yolov10l)  [`yolov10x`](https://huggingface.co/jameslahm/yolov10x)  
```
yolo val model=jameslahm/yolov10{n/s/m/b/l/x} data=coco.yaml batch=256
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.val(data='coco.yaml', batch=256)
```


## Training 
```
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='coco.yaml', epochs=500, batch=256, imgsz=640)
```

## Push to hub to 🤗

Optionally, you can push your fine-tuned model to the [Hugging Face hub](https://huggingface.co/) as a public or private model:

```python
# let's say you have fine-tuned a model for crop detection
model.push_to_hub("<your-hf-username-or-organization/yolov10-finetuned-crop-detection")

# you can also pass `private=True` if you don't want everyone to see your model
model.push_to_hub("<your-hf-username-or-organization/yolov10-finetuned-crop-detection", private=True)
```

## Prediction
Note that a smaller confidence threshold can be set to detect smaller objects or objects in the distance. Please refer to [here](https://github.com/THU-MIG/yolov10/issues/136) for details.
```
yolo predict model=jameslahm/yolov10{n/s/m/b/l/x}
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.predict()
```

## Export
```
# End-to-End ONNX
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=onnx opset=13 simplify
# Predict with ONNX
yolo predict model=yolov10n/s/m/b/l/x.onnx

# End-to-End TensorRT
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=engine half=True simplify opset=13 workspace=16
# or
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# Predict with TensorRT
yolo predict model=yolov10n/s/m/b/l/x.engine
```

Or
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.export(...)
```

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics), [RT-DETR](https://github.com/lyuwenyu/RT-DETR) and YOLOv10.

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
# 
# 
