# Pytorch_Quantization_EX (on progress)
- Code Refactoring 

### 0. Introduction
- Goal : Quantization Model PTQ & QAT
- Process : 
  1. Pytorch model train with custom dataset 
  2. Pytorch-Quantization model calibration for ptq
  3. Pytorch-Quantization model fine tuning for qat
  4. Generation TensorRT int8 model from Pytorch-Quantization model
  4. Generation TensorRT int8 model using tensorrt calibration class
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
    ├── common.py           # utils for TensorRT
    ├── infer.py            # base model infer
    ├── ptq.py              # Post Train Quantization
    ├── quant_utils.py      # utils for quantization
    ├── qat.py              # Quantization Aware Training
    ├── train.py            # base model train
    ├── trt_infer.py        # TensorRT model infer
    ├── utils.py            # utils
    ├── LICENSE
    └── README.md
```

---

### 3. Performance Evaluation
- Calculation 10000 iteration with one input data [1, 3, 224, 224]
<!-- 
<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td>
            <td><strong>Pytorch Quantization</strong></td>
            <td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td>
            <td>FP32</td>
            <td>Int8</td>
            <td>Int8</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>  0.4337 ms </td>
			<td>  0.3145 ms </td>
			<td>  0.0077 ms </td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td> 2305 fps </td>
			<td> 3179 fps </td>
			<td> 129620 fps </td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>   GB </td>
			<td>   GB </td> 
			<td>  0.648 GB </td>
		</tr>
	</tbody>
</table> -->

### 4. Guide
- infer -> train -> ptq -> qat -> trt_infer

### 5. Reference
* pytorch-quantization : <https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization>
* imagenet100 : <https://www.kaggle.com/datasets/ambityga/imagenet100>

