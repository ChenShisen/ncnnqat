<div id="ncnnqat"></div>

# ncnnqat

ncnnqat is a quantize aware training package for NCNN on pytorch.

<div id="table-of-contents"></div>

## Table of Contents

- [ncnnqat](#ncnnqat)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Code Examples](#code-examples)
  - [Results](#results)
  - [Todo](#todo)


<div id="installation"></div>  

## Installation

* Supported Platforms: Linux
* Accelerators and GPUs: NVIDIA GPUs via CUDA driver ***10.1***.
* Dependencies:
  * python >= 3.5, < 4
  * pytorch >= 1.6
  * numpy >= 1.18.1
  * onnx >= 1.7.0
  * onnx-simplifier >= 0.3.5

* Install ncnnqat via pypi:  
  ```shell
  $ pip install ncnnqat (to do....)
  ```
  It is recommended to install from the source code
* or Install ncnnqat via repo：
  ```shell
  $ git clone https://github.com/ChenShisen/ncnnqat
  $ cd ncnnqat
  $ make install
  ```

<div id="usage"></div>

## Usage


* merge bn weight into conv and freeze bn

  suggest finetuning from a well-trained model, register_quantization_hook and merge_freeze_bn at beginning. do it after a few epochs of training otherwise.

  ```python
  from ncnnqat import quant_dequant_weight, unquant_weight, merge_freeze_bn, register_quantization_hook
  ...
  ...
      for epoch in range(epoch_train):
		  model.train()
		  if epoch==well_epoch:
			  register_quantization_hook(model)
		  if epoch>=well_epoch:
			  model = merge_freeze_bn(model)  #it will change bn to eval() mode during training
  ...
  ```

* Unquantize weight before update it

  ```python
  ...
  ...
      model.apply(unquant_weight)  # using original weight while updating
      optimizer.step()
  ...
  ```

* Save weight and save ncnn quantize table after train


  ```python
  ...
  ...
      onnx_path = "./xxx/model.onnx"
	  table_path="./xxx/model.table"
	  dummy_input = torch.randn(1, 3, img_size, img_size, device='cuda')
      input_names = [ "input" ]
      output_names = [ "fc" ]
      torch.onnx.export(model, dummy_input, onnx_path, verbose=False, input_names=input_names, output_names=output_names)
	  save_table(model,onnx_path=onnx_path,table=table_path)

  ...
  ```
  if use "model = nn.DataParallel(model)",pytorch unsupport torch.onnx.export,you should save state_dict first and  prepare a new model with one gpu,then you will export onnx model.
  
  ```python
  ...
  ...
      model_s = new_net() #
	  model_s.cuda()
	  register_quantization_hook(model_s)
	  #model_s = merge_freeze_bn(model_s)
      onnx_path = "./xxx/model.onnx"
	  table_path="./xxx/model.table"
	  dummy_input = torch.randn(1, 3, img_size, img_size, device='cuda')
      input_names = [ "input" ]
      output_names = [ "fc" ]
	  model_s.load_state_dict({k.replace('module.',''):v for k,v in model.state_dict().items()}) #model_s = model     model = nn.DataParallel(model)
            
      torch.onnx.export(model_s, dummy_input, onnx_path, verbose=False, input_names=input_names, output_names=output_names)
	  save_table(model_s,onnx_path=onnx_path,table=table_path)
	  

  ...
  ```

* Using EMA with caution(Not recommended).

<div id="code-examples"></div>

## Code Examples

  Cifar10 quantization aware training example.

  ```python test/test_cifar10.py```

<div id="results"></div>

## Results  

* Cifar10


  result：

    |  net   | fp32(onnx) | ncnnqat     | ncnn aciq     | ncnn kl |
    | -------- |  -------- | -------- | -------- | -------- |
    | mobilenet_v2     | 0.91  | 0.9066  | 0.9033 | 0.9066 |
    | resnet18 | 0.94   | 0.93333   | 0.9367 | 0.937|


* coco

  ....


<div id="todo"></div>

## Todo

   ....
