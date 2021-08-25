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
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata

#logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("compile_engine")
#logging.getLogger("compile_engine").setLevel(logging.INFO)
#logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

core_type = -1
core_num  = 2
# get the target device type 
target_type  = sys.argv[1]
#network_name = sys.argv[2]
#network_name = "mnist"
#network_name = "mobilenet"
#network_name = "op9_dla"
network_name = "op9_dla_tflite"
# load the nn model
if network_name == "mnist":
    onnx_model = onnx.load('mnist/mnist-8.onnx')
    # configure the model input shape
    input_name = "Input3"
    input_shape = (1, 1, 28, 28);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "mobilenet":
    onnx_model = onnx.load('mobilenet/onnx/mobilenetv2-7.onnx')
    input_name = "input"
    input_shape = (1, 3, 224, 224);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "op9_dla":
    onnx_model = onnx.load('op9_dla/op9_dla.onnx')
    input_name = "input.1"
    input_shape = (1, 320, 40, 24);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
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
    model_dir = "op9_dla"
    #tflite_model_file = os.path.join(model_dir, "op9_dla_fp32.tflite")
    tflite_model_file = os.path.join(model_dir, "op9_dla_int8.tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    input_name = "serving_default_input.1:0"
    input_shape = (1, 320, 40, 24);
    #input_dtype = "float32"
    input_dtype = "int8"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    model, params = relay.frontend.from_tflite(tflite_model, shape_dict, dtype_dict)

    # Configure the quantization behavior
    """
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
    """
# The following is my environment, change this to the IP address of
# your target device
host = "192.168.104.240"
port = 9090

if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
    target_host = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target      = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
elif target_type == 'opencl':
    #target      = tvm.target.Target("opencl -device=powervr")
    target      = tvm.target.Target("opencl -device=powervr -model=v9h -max_num_threads=512 -thread_warp_size=1")
    #target      = tvm.target.Target("opencl -device=mali")
    #target      = tvm.target.Target("opencl -device=intel_graphics")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
print("device:", target)
print("host:", target_host)

with tvm.transform.PassContext(opt_level=3):
    model_lib = relay.build(model, target=target, target_host=target_host, params=params)
    #model_lib = relay.build(quan_model, target=target, target_host=target_host, params=params)
print(model_lib)

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
model_lib.export_library(lib_fname)
#exp_lib_path = "./" + "aarch64_deploy_lib.so"
#model_lib.export_library(exp_lib_path,cc="/home/guohua.zhu/Workspace/toolchains/gcc-7.5.0-aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")

if target_type == 'x86_64':
    remote = rpc.LocalSession()
    dev = remote.cpu(0)
elif target_type == 'aarch64':
    remote = rpc.connect(host, port)
    dev = remote.cpu(0)
elif target_type == 'opencl':
    remote = rpc.connect(host, port)
    dev = remote.cl(0)

print("Target type:", target_type)
config_func = remote.get_function('runtime.config_threadpool')
config_func(core_type, core_num)
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
    # run
    module.run()

    # get output
    tvm_output = module.get_output(0).numpy()
    top1 = np.argmax(tvm_output[0])
    print("top1:", top1)

    #label_path = download_testdata(label_file_url, label_file, module="data")

    # List of 1001 classes
    #with open(label_path) as f:
    #    labels = f.readlines()

    # Convert result to 1D data
    #predictions = np.squeeze(tvm_output)

    # Get top 1 prediction
    #prediction = np.argmax(predictions)

    # Convert id to class name and show the result
    #print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
else:
    # set input data
    module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))) 
# get the result
#number = np.argmax(out.numpy())
#print("TVM prediction number: ", number)

module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))) 
# evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, number=50, repeat=3)
prof_res = np.array(ftimer().results) * 1000
# convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
