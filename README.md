# Training YOLOv5 and v10 
This repo contains the code required to train, validate, quantize and convert YOLO models v5 and v8-10. 


## Training
### Preparing Dataset
1. Download dataset of images and labels
2. Ensure images and labels are organised in the following way: 
```
dataset/
├── train
│   ├── images
│   └── labels
└── val
    ├── images
    └── labels
```
3. This repo already contains a data.yaml file; ensure that the paths, number of classes, and class names match your data set.
### Training
1. Create  new anaconda environment with Python 3.9. Follow [these instructions](https://stackoverflow.com/a/63306300) to create an environment that uses GPU. 
2. Run the first code cell of `train_v5_v10.ipynb` to install the requirements
3. To train YOLOv5, run the following code cell (weights can be modified depending on which variant of YOLOv5 you want to train, they will be downloaded automatically):
```
!python yolov5/train.py --imgsz 480  --epochs 300  --batch 30  --patience 30  --data data.yaml --weights yolov5n6.pt
```

4. To train YOLOv8 to v10, run the following code (again, change the model argument to train a different version/size of YOLO):
```
!yolo task=detect mode=train epochs=300 patience=30 batch=30 plots=True model=yolov10n.pt data=data.yaml imgsz=480
```
** imgsz, patience, batch and plots are optional arguments. 

## Validation and Inference
Use ```validation.ipynb``` to run validation. The trained v5n6, v10n and v10s weights for DvD are already provided in OpenVino format. 

### YOLO v5
```
!python yolov5/val.py --weights "trained_weights/drone_cpu_quantized_openvino_model"  --data data.yaml --img 480
```
### YOLO v8 - v10
```
from ultralytics import YOLO
model  = YOLO('trained_weights/v10n_openvino_model')
model.val(data='data_training_data.yaml', imgsz=480)
```

## Conversion and Quantization
### OpenVINO Format
To run YOLO on a small Intel CPU, it is best that the weights are converted to OpenVINO format. 
### What is Quantization
Most neural networks make use of 32-bit floating point numbers to do the calculations and operations. While precise, this is also computationally expensive. One way to speed things up would be through quantization, where we round these numbers to 8-bit integers instead.
### convert_quantize_weights.ipynb
To quantize the weights and convert them into OpenVINO, we first need to export the pytorch (.pt) weights to .onnx
```
# v8-v10
from ultralytics import YOLO
model10n  = YOLO('trained_weights/v10n.pt')
model10n.export(format='onnx', imgsz=480)
```
```
#v5
!python yolov5/export.py --weights yolov5n6.pt --img 480 --include "onnx" 
```
Then we quantize it with Neural Network Compression Framework (NNCF) 
```
import  nncf
import openvino.runtime as  ov
import  torch
from torchvision import datasets, transforms
import openvino as  ov

# Instantiate your uncompressed model
model10n  =  ov.Core().read_model("trained_weights/v10n.onnx")

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset  = datasets.ImageFolder("train_val_data/", transform=transforms.Compose([transforms.RandomResizedCrop(480),transforms.ToTensor()]))
dataset_loader  =  torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function

def  transform_fn(data_item):
	images, _  =  data_item
	return  images

# Step 2: Initialize NNCF Dataset
calibration_dataset  =  nncf.Dataset(dataset_loader, transform_fn)

# Step 3: Run the quantization pipeline and save the model
quantized_model10n  =  nncf.quantize(model10n, calibration_dataset)
ov.save_model(quantized_model10n, 'trained_weights/v10n_quantized_openvino_model/v10n_quantized.xml')

```

