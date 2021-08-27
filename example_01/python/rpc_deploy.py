#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import logging
from PIL import Image
import sys
import numpy as np
import onnx
import os
import glob
from onnx import numpy_helper

from google.protobuf import text_format
#from tvm.relay.frontend import caffe_pb2 as pb
#import caffe
#from caffe import layers as L, params as P
#from caffe.proto import caffe_pb2 as pb

import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# The following is my environment, change this to the IP address of
# your target device
host = "192.168.104.240"
port = 9090
# selecrt whether model quantify
quantify_enable = True
# configure the cpu cores
core_type = -1
core_num  = 6
# get the target device type 
target_type  = sys.argv[1]
network_name = sys.argv[2]
#network_name = "jreg_caffe"
#network_name = "jreg_onnx"
network_name = "jseg_caffe"
#network_name = "yfast_onnx"

pretrained_model_path = "../pretrained_models"
# load the nn model
if network_name == "mnist":
    model_path = os.path.join(pretrained_model_path, "mnist/mnist-8.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name  = "Input3"
    input_shape = (1, 1, 28, 28);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "mobilenet":
    model_path = os.path.join(pretrained_model_path, "mobilenet/onnx/mobilenetv2-7.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input"
    input_shape = (1, 3, 224, 224);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "op9_dla_onnx":
    model_path = os.path.join(pretrained_model_path, "op9_dla/onnx/op9_dla.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input.1"
    input_shape = (1, 320, 40, 24);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)

    if quantify_enable:
        # Configure the quantization behavior
        qconfig = relay.quantize.qconfig(skip_conv_layers=[0],
                        nbit_input=8,
                        nbit_weight=8,
                        global_scale=8.0,
                        dtype_input='int8',
                        dtype_weight='int8',
                        dtype_activation='int8',
                        debug_enabled_ops=None)
        with qconfig:
            quan_model = relay.quantize.quantize(model, params)
elif network_name == "op9_dla_tflite":
    # Now we can open mobilenet_v1_1.0_224.tflite
    model_path = os.path.join(pretrained_model_path, "op9_dla/tflite/op9_dla_int8.tflite")
    #model_path = os.path.join(pretrained_model_path, "op9_dla/tflite/op9_dla_fp32.tflite")
    tflite_model_buf = open(model_path, "rb").read()
    # Get TFLite model from buffer
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    # configure the model input ,shape ans type
    input_name = "serving_default_input.1:0"
    input_shape = (1, 320, 40, 24);
    input_dtype = "int8"
    #input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_tflite(tflite_model, shape_dict, dtype_dict)

elif network_name == "jreg_caffe":
    # configure the model net
    init_net = os.path.join(pretrained_model_path, "jreg_288hx240w/caffemodel/jreg.caffemodel")
    predict_net = os.path.join(pretrained_model_path, "jreg_288hx240w/caffemodel/jreg_deploy.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (1, 3, 288, 240);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_caffe2(init_net, predict_net, shape_dict, dtype_dict)

elif network_name == "jreg_onnx":
    model_path = os.path.join(pretrained_model_path, "jreg_288hx240w/onnx/JSeg_optset11.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input.1"
    input_shape = (1, 3, 288, 240);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "jseg_caffe":
    # configure the model net
    init_net = os.path.join(pretrained_model_path, "jseg_224hx256w/jseg.caffemodel")
    predict_net = os.path.join(pretrained_model_path, "jseg_224hx256w/deploy_256x224.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (1, 3, 224, 256);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_caffe2(init_net, predict_net, shape_dict, dtype_dict)
elif network_name == "yfast_onnx":
    model_path = os.path.join(pretrained_model_path, "yfast_128hx128w/yfast_128x128.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input"
    input_shape = (1, 3, 128, 128);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)


if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
    target_host = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target      = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
elif target_type == 'opencl':
    target      = tvm.target.Target("opencl -device=mali")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
print("device:", target)
print("host:", target_host)

with tvm.transform.PassContext(opt_level=3):
    model_lib = relay.build(model, target=target, target_host=target_host, params=params)
print(model_lib)

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
model_lib.export_library(lib_fname)

if target_type == 'x86_64':
    remote = rpc.LocalSession()
    dev = remote.cpu(0)
elif target_type == 'aarch64':
    remote = rpc.connect(host, port)
    dev = remote.cpu(0)
    # configure the number of cpu running cores
    config_func = remote.get_function('runtime.config_threadpool')
    config_func(core_type, core_num)
elif target_type == 'opencl':
    remote = rpc.connect(host, port)
    dev = remote.cl(0)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.tar")
module = runtime.GraphModule(rlib["default"](dev))

if network_name == "mobilenet":
    img_path = "data/cat.png"
    img = Image.open(img_path).resize((224, 224))
    def transform_image(image):
        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image
    image_data = transform_image(img)
    print("input", image_data.shape)
    module.set_input(input_name, tvm.nd.array(image_data.astype(input_dtype))) 
    module.run()
    tvm_output = module.get_output(0).numpy()
    top1 = np.argmax(tvm_output[0])
    print("mobile top1:", top1)
elif network_name == "mnist": 
    # set input data
    module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))) 
    # run
    module.run()
    # get the result
    number = np.argmax(module.get_output(0).numpy())
    print("mnist prediction number: ", number)

# evaluate
module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))) 
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, number=50, repeat=3)
# convert to millisecond
prof_res = np.array(ftimer().results) * 1000
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
