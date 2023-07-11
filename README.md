# Pytorch_Quantization_EX (on progress)
- Code Refactoring 

### 0. Introduction
- Goal : Quantization Model PTQ & QAT
- Process : 
  1. Pytorch model (model train) 
  2. Pytorch Quantization model (calibration), ptq
  3. Pytorch Quantization model (fine tuning), qat
  4. Generation TensorRT int8 model
- Sample Model : Resnet18 
- Dataset : imagenet100
---

### 1. Development Environment
- Device 
  - Windows 10 laptop
  - CPU i7-11375H
  - GPU RTX-3060
- Dependency 
  - cuda 11.4.1
  - cudnn 8.4.1
  - tensorrt 8.4.3.1
  - pytorch 1.13.1+cu117
  - onnx 1.13.1

---

### 2. Code Scheme
```
    Pytorch_Quantization_EX/
    ├── data/                         # binary input data & outputs data
    ├── python/
    │   ├─ data/                      # for cifar10 dataset
    │   ├─ model/                     # onnx, pth, wts files
    │   ├─ 1_resnet18_train.py        # resnet18 cifar10 data train code
    │   ├─ 2_resnet18_ptq.py
    │   ├─ 3_resnet18_qat.py
    │   ├─ 4_onnx_export.py           # make onnx for TRT
    │   ├─ 5_valid.py
    │   └─ utils.py                   # custom util functions
    ├── TensorRT_ONNX/ 
    │   ├─ Engine/                    # engine file 
    │   ├─ TensorRT_ONNX/
    │   │   ├─ calibrator.cpp         # for ptq
    │   │   ├─ calibrator.hpp
    │   │   ├─ logging.hpp
    │   │   ├─ main.cpp               # main code
    │   │   ├─ utils.cpp              # custom util functions
    │   │   └─ utils.hpp
    │   └─ TensorRT_ONNX.sln
    ├── LICENSE
    └── README.md
```

---

### 3. Performance Evaluation
- Comparison of calculation average execution time of 40 iteration and FPS, GPU memory usage for input [256,3,32,32]
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


https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization