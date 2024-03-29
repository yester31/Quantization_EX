# Pytorch_Quantization_EX 


### 0. Introduction
- Goal : Quantization Model PTQ & QAT
- Process : 
  1. Pytorch model train with custom dataset 
  2. Pytorch-Quantization model calibration for ptq
  3. Pytorch-Quantization model fine tuning for qat
  4. Generation TensorRT int8 model from Pytorch-Quantization model
  5. Generation TensorRT int8 model using tensorrt calibration class
- Sample Model : Resnet18 
- Dataset : imagenet100
---

### 1. Development Environment
- Device 
  - Windows 10 laptop
  - CPU i7-11375H
  - GPU RTX-3060
- Dependency 
  - cuda 12.1
  - cudnn 8.9.2
  - tensorrt 8.6.1
  - pytorch 2.1.0+cu121

---

### 2. Code Scheme
```
    Quantization_EX/
    ├── calibrator.py       # calibration class for TensorRT PTQ
    ├── common.py           # utils for TensorRT
    ├── infer.py            # base model infer
    ├── onnx_export.py      # onnx export
    ├── ptq.py              # Post Train Quantization
    ├── qat.py              # Quantization Aware Training
    ├── quant_utils.py      # utils for quantization
    ├── train.py            # base model train
    ├── trt_infer.py        # TensorRT model infer
    ├── utils.py            # utils
    ├── LICENSE
    └── README.md
```

---

### 3. Performance Evaluation
- Calculation 10000 iteration with one input data [1, 3, 224, 224]

<table border="0"  width="100%">
    <tbody align="center">
        <tr>
            <td></td>
            <td><strong>TRT</strong></td>
            <td><strong>TRT</strong></td>
            <td><strong>TRT PTQ</strong></td>
            <td><strong>PT-Q PTQ</strong></td>
            <td><strong>PT-Q PTQ w bnf </strong></td>
            <td><strong>PT-Q QAT</strong></td>
            <td><strong>PT-Q QAT w bnf </strong></td>
        </tr>
        <tr>
            <td>Precision</td>
            <td>FP32</td>
            <td>FP16</td>
            <td>Int8</td>
            <td>Int8</td>
            <td>Int8</td>
            <td>Int8</td>
            <td>Int8</td>
        </tr>
        <tr>
            <td>Acc Top-1 [%] </td>
            <td>  83.08  </td>
            <td>  83.04  </td>
            <td>  83.12  </td>
            <td>  83.18  </td>
            <td>  82.64  </td>
            <td>  83.42  </td>
            <td>  82.80  </td>
        </tr>
        <tr>
            <td>Avg Latency [ms]</td>
            <td>  1.188 ms </td>
            <td>  0.527 ms </td>
            <td>  0.418 ms </td>
            <td>  0.566 ms </td>
            <td>  0.545 ms </td>
            <td>  0.577 ms </td>
            <td>  0.534 ms </td>
        </tr>
        <tr>
            <td>Avg FPS [frame/sec]</td>
            <td> 841.74 fps </td>
            <td> 1896.01 fps </td>
            <td> 2388.33 fps </td>
            <td> 1764.55 fps </td>
            <td> 1834.69 fps </td>
            <td> 1730.89 fps </td>
            <td> 1870.99 fps </td>
        </tr>
        <tr>
            <td>Gpu Memory [MB]</td>
            <td>  179 MB </td>
            <td>  135 MB </td> 
            <td>  123 MB </td>
            <td>  129 MB </td>
            <td>  129 MB </td>
            <td>  129 MB </td>
            <td>  129 MB </td>
        </tr>
    </tbody>
</table>

- PT : Ptorch Quantization    
- TRT : TensorRT
- bnf : bach normalization folding (conv + bn -> conv')
### 4. Guide
- infer -> train -> ptq -> qat -> onnx_export -> trt_infer -> trt_infer_acc

### 5. Reference
* pytorch-quantization : <https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization>
* imagenet100 : <https://www.kaggle.com/datasets/ambityga/imagenet100>

