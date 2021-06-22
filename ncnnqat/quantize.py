#!/usr/bin/env python

import logging
import torch
import numpy as np
import onnx

from quant_cuda import fake_quantize

class FakeQuantCuda():
    r"""
    """
    def __init__(self,
                 bit_width=8,
                 type=1,
                 c=1
                 ):
        
        self._bit_width = bit_width
        self._type = type
        self._c = c
        

    def __call__(self, tensor,tensor_scale,tensor_movMax=None, aciq=0): #type=0,1,2=pre_conv_activate,w,after_conv_activate  
        r""" Converts float weights to quantized weights.

        Args:
            - tensor: input data
            - tensor_scale data scale data
            - tensor_movMax tensor max value 
            - aciq qat methed ,default turn of, use kl
        """
        
        #print(self._type,self._bit_width)
        #tensor.data = fake_quantize_c(tensor.data.detach().clone(),tensor_s.data.detach().clone(),self._bit_width,self._type)
        
        out = fake_quantize(tensor.data.detach().clone(),self._bit_width,self._type,self._c,aciq)
        tensor.data = out[0]
        tensor_scale.data = out[1]
        if self._type==0:
            tensor_movMax.data = out[2]
        #print("tensor_scale",tensor_scale)
            
        return tensor,tensor_scale,tensor_movMax




def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """ fuse convolution and batch norm's weight.

    Args:
        conv_w (torch.nn.Parameter): convolution weight.
        conv_b (torch.nn.Parameter): convolution bias.
        bn_rm (torch.nn.Parameter): batch norm running mean.
        bn_rv (torch.nn.Parameter): batch norm running variance.
        bn_eps (torch.nn.Parameter): batch norm epsilon.
        bn_w (torch.nn.Parameter): batch norm weight.
        bn_b (torch.nn.Parameter): batch norm weight.

    Returns:
        conv_w(torch.nn.Parameter): fused convolution weight.
        conv_b(torch.nn.Parameter): fused convllution bias.
    """

    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * \
        (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def _fuse_conv_bn(conv, bn):
    conv.weight, conv.bias = \
        _fuse_conv_bn_weights(conv.weight, conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return conv


def _fuse_modules(model):
    r"""Fuses a list of modules into a single module

    Fuses only the following sequence of modules:
    conv, bn
    All other sequences are left unchanged.
    For these sequences, fuse modules on weight level, keep model structure unchanged.

    Arguments:
        model: Model containing the modules to be fused

    Returns:
        model with fused modules.

    """
    children = list(model.named_children())
    conv_module = None
    conv_name = None

    for name, child in children:
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                              torch.nn.BatchNorm3d)):
            if isinstance(conv_module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                conv_module = _fuse_conv_bn(conv_module, child)
                model._modules[conv_name] = conv_module
                child.eval()
                child.running_mean = child.running_mean.new_full(
                    child.running_mean.shape, 0)
                child.running_var = child.running_var.new_full(
                    child.running_var.shape, 1)
                
                if child.weight is not None:
                    child.weight.data = child.weight.data.new_full(
                        child.weight.shape, 1)
                if child.bias is not None:
                    child.bias.data = child.bias.data.new_full(
                        child.bias.shape, 0)
                #print(child,child.bias)
                child.track_running_stats = False
                child.momentum = 0
                child.eps = 0
                #child.affine  = False
            conv_module = None
        elif isinstance(child, (torch.nn.Conv2d, torch.nn.Conv3d)):
            conv_module = child
            conv_name = name
        else:
            _fuse_modules(child)
    return model


def freeze_bn(m, freeze_bn_affine=True):
    """Freeze batch normalization.
        reference: https://arxiv.org/abs/1806.08342


    Args:
        - m (nn.module): torch module
        - freeze_bn_affine (bool, optional): Freeze affine scale and
        translation factor or not. Defaults: True.
    """

    if isinstance(
            m,
        (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):

        m.eval()
        if freeze_bn_affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def merge_freeze_bn(model):
    """merge batch norm's weight into convolution, then freeze it.

    Args:
        model (nn.module): model.

    Returns:
        [nn.module]: model.
    """
    model = _fuse_modules(model) #merge conv bn ; mean 0 std 1 gama 1 beta 0 
    model.apply(freeze_bn) # bn backward = false,bn not train 
    return model


def unquant_weight(m):
    """ unquantize weight before update weight, avoid training turbulence.

    Args:
        - m (nn.module): torch module.
    """
    try:
        if hasattr(m, "weight_origin") and m.weight is not None:
            m.weight.data.copy_(m.weight_origin.data)
    except AttributeError:
        pass
    except TypeError:
        pass

'''
def quant_dequant_weight(m):
    """ quant weight manually.

    Args:
        - m (nn.module): torch module.
    """
    quant_handle = FakeQuantCuda()
    try:
        if hasattr(m, "weight_origin") and m.weight is not None:
            m.weight_origin.data.copy_(m.weight.data)
            m.weight.data = quant_handle(m.weight.data.detach().clone())
    except AttributeError:
        pass
    except TypeError:
        pass
'''

def _quantizing_activation_ncnn(module, input):
    """ quantize per-layer activation(input of layer) before layer calculate.

    Args:
        - module (nn.module): torch module.
        - input : layer input(tuple) ,torch tensor (nchw or n**).
    """
    #GOOGLE QAT  movMax = movMax*momenta + max(abs(tensor))*(1-momenta)    momenta = 0.95
    #print("input.shape",input[0].shape)
    aciq = 0
    quant_handle = FakeQuantCuda(type=0,bit_width=8,c=1)
    if isinstance(input, tuple):
        for item in input:
            aciq = 0
            item_type = item.dtype
            if item.numel()/item.shape[0]>8000:
                aciq = 1
            #quant_tuple = quant_handle(item.float(),module.activation_scale.data.detach().clone())
            quant_tuple = quant_handle(item,module.activation_scale.data.detach().clone(),tensor_movMax=module.activation_movMax.data.detach().clone(),aciq=aciq)
            item = quant_tuple[0]
            if item.dtype!=item_type:
                #print(item.dtype,item_type)
                item.to(item_type)
            module.activation_scale.data = quant_tuple[1]
            module.activation_movMax.data = quant_tuple[2]
            #print(quant_tuple[2])

    else:
        if input.numel()/input.shape[0]>8000:
            aciq = 1
        #quant_tuple = quant_handle(input.float(),module.activation_scale.data.detach().clone())
        quant_tuple = quant_handle(input,module.activation_scale.data.detach().clone(),tensor_movMax=module.activation_movMax.data.detach().clone(),aciq=aciq)
        input = quant_tuple[0]
        module.activation_scale.data = quant_tuple[1]
        module.activation_movMax.data = quant_tuple[2]
def _quantizing_weight_ncnn(module, input):
    """ quantize per-channel weight before layer calculate.

    Args:
        - module (nn.module): torch module.
        - input : layer input(tuple) ,torch tensor (nchw or n**).
    """
    module_shape = module.weight.shape
    #print("module_shape",module_shape)
    channel = module_shape[0] #oikk
    if isinstance(module,(torch.nn.Conv2d)) and module.groups!=1:   #depthwise
        channel = module.groups
    bit_width = 8
    if isinstance(module,(torch.nn.Conv2d)) and module.stride==(1,1) and module.dilation==(1,1) and module.kernel_size==(3,3) and module.groups==1: #winnograd f(4,3)
        bit_width=6
        
    aciq = 0    
    weight_numel = module.weight.numel()
    if weight_numel/channel>8000: #when > 8000 , max_var > threshold
        aciq = 1
        #print("aciq",aciq,module)
    

    quant_handle = FakeQuantCuda(type=1,bit_width=bit_width,c=channel)
    # print("quantizing weight.")
    # print(module.weight[0][0][0])
    module.weight_origin.data.copy_(module.weight.data) #copy float data to a new place
    
    quant_tuple = quant_handle(module.weight.data.detach().clone(),module.weight_scale.data.detach().clone(),aciq=aciq)#把原始数据 quant——dequant 此时数据是有损的，计算损失后，把备份数据考回原处做梯度计算
    module.weight.data = quant_tuple[0]
    module.weight_scale.data = quant_tuple[1]
    # print(module.weight[0][0][0])
    #print(module.weight_scale)


def register_quantization_hook(model,
                               quant_weight=True,
                               quant_activation=True,
                              ):
    """register quantization hook for model.

    Args:
        model (:class:`Module`): Module.

    Returns:
        Module: self
    """

    #  weight quantizing.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for _, module in model._modules.items():
        #print("module",module)
        if len(list(module.children())) > 0:
            register_quantization_hook(module, quant_weight, quant_activation)
        else:
            if quant_weight and hasattr(module,"weight") and module.weight is not None and isinstance(
                    module, (torch.nn.Conv2d,torch.nn.Linear)):
                module.register_buffer('weight_origin', module.weight.detach().clone()) #数据备份空间
                #module.register_buffer("weight_scale", torch.ones([1,model._modules["conv1"].weight.shape[0]], dtype=torch.float).cuda()) #weight scale
                #module.register_buffer("weight_scale", torch.ones([1,module.weight.shape[0]], dtype=torch.float).cuda()) #weight scale module.weight.shape =[o,i,k,k]
                module.register_buffer("weight_scale", torch.ones([module.weight.shape[0]], dtype=torch.float).cuda()) #weight scale module.weight.shape =[o,i,k,k]

      
                module.register_forward_pre_hook(_quantizing_weight_ncnn)
                logger.info("Quantizing weight of %s", str(module))
            
            
                module.register_buffer("activation_scale", torch.tensor([1], dtype=torch.float).cuda())
                module.register_buffer("activation_movMax", torch.tensor([1], dtype=torch.float).cuda())
                #module.register_buffer("activation_momenta", torch.tensor([1], dtype=torch.float).cuda())
                module.register_forward_pre_hook(_quantizing_activation_ncnn)
                logger.info("Quantizing activation of %s", str(module))

    return model
    
def save_table(torch_model,onnx_path="model.onnx",table="model.table"):
    f = open(table,"w",encoding='utf8')
    static_dict_org = torch_model.state_dict()
    static_dict = {k.replace('module.',''):v for k,v in static_dict_org.items()}
    
    
    model = onnx.load(onnx_path)
    node = model.graph.node
    node_num = len(node)
    
    tail_layer = "_param_0"
    split_char = " "
    tab_char = "\n"
    tail_len = 6
    for each in range(node_num):
        if node[each].op_type not in ["Conv","Gemm"]:
            continue
        #print(node[each].op_type)
        pre_name = node[each].input[1]
        #print(pre_name)
        #print(pre_name.replace(pre_name.split(".")[-1],"weight_scale"))
        scale_data = static_dict[pre_name.replace(pre_name.split(".")[-1],"weight_scale")]
        list_scale = scale_data.cpu().numpy().flatten().tolist()
        #print(node[each].name,node[each].op_type,node[each].input)
        f.write(node[each].name + tail_layer)
        for d in list_scale:
            d = float(d)
            f.write(split_char + "{:.6f}".format(d))
        f.write(tab_char)
    for each in range(node_num):
        if node[each].op_type not in ["Conv","Gemm"]:
            continue
        pre_name = node[each].input[1]
        scale_data = static_dict[pre_name.replace(pre_name.split(".")[-1],"activation_scale")]
        list_scale = scale_data.cpu().numpy().flatten().tolist()
        #print(node[each].name,node[each].op_type,node[each].input)
        f.write(node[each].name)
        for d in list_scale:
            d = float(d)
            f.write(split_char + "{:.6f}".format(d))
            
        f.write(tab_char)
    f.close()
