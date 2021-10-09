#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import sys
import numpy as np
import onnx
import os
import glob
from onnx import numpy_helper
import tvm
from tvm import te
import tvm.relay as relay

target_type = sys.argv[1]

network_name = "mnist"
network_name = "mobilenet"

batch_size = 1
pretrained_model_path = "../pretrained_models"
auto_schedule_log_path = "../AutoSchedule_log_files"
model_path = "%s/%s" % (auto_schedule_log_path, network_name)
# set the name of log file
history_log_file = "%s-%s-history-log.json" % (network_name, target_type)
best_log_file = "%s-%s-history-log.json.best.json" % (network_name, target_type)

history_log_path = os.path.join(model_path, history_log_file)
best_log_path = os.path.join(model_path, best_log_file)

# load the nn model
if network_name == "mnist":
    model_path = os.path.join(pretrained_model_path, "mnist/mnist-8.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name  = "Input3"
    input_shape = (batch_size, 1, 28, 28);
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
    input_shape = (batch_size, 3, 224, 224);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "resnet":
    model_path = os.path.join(pretrained_model_path, "resnet/onnx/resnet50-v2-7.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (batch_size, 3, 224, 224);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)

"""
mnist_model = onnx.load('../pretrained_models/mnist/mnist-8.onnx')

input_name = "Input3"
shape_dict = {input_name: (1, 1, 28, 28)}
print(shape_dict)

mod, params = relay.frontend.from_onnx(mnist_model, shape_dict)
"""

if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
    target_host = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target      = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
elif target_type == 'opencl':
    target      = tvm.target.Target("opencl -max_num_threads=512")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
                                        
print(target)
print(target_host)
if os.path.exists(best_log_path):
    print("Compile Loaded File:", best_log_path)
    with auto_scheduler.ApplyHistoryBest(best_log_path):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            model_lib = relay.build(model, target=target, target_host=target_host, params=params)
elif os.path.exists(history_log_path):
    print("Compile Loaded File:", history_log_path)
    with auto_scheduler.ApplyHistoryBest(history_log_path):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            model_lib = relay.build(model, target=target, target_host=target_host, params=params)
else: 
    print("Compile Without Auto Scheduler")
    with tvm.transform.PassContext(opt_level=3):
        model_lib = relay.build(model, target=target, target_host=target_host, params=params)

"""
with tvm.transform.PassContext(opt_level=3):
    module_lib = relay.build(mod, target=target, target_host=target_host, params=params, mod_name='mnist')
print(module_lib)
"""

# lib export
lib_file = "%s.so" % (network_name)
param_file = "%s.params" % (network_name)
model_path = "models/%s" % (target_type)

if not os.path.exists(model_path):
    os.system("mkdir -p %s" % (model_path))
"""
if not os.path.exists(model_path):
    os.system("mkdir -p %s" % (model_path))
"""

lib_path   = os.path.join(model_path, lib_file)
param_path = os.path.join(model_path, param_file)

if target_type == 'x86_64':
    model_lib.export_library(lib_path)
else:
    model_lib.export_library(lib_path, tvm.contrib.cc.cross_compiler('aarch64-linux-gnu-g++'))
"""
if target_type == 'x86_64':
    lib_path   = "models/x86_64/mnist.so"
    param_path = "models/x86_64/mnist.params"
    module_lib.export_library(lib_path)
elif target_type == 'aarch64':
    lib_path   = "models/aarch64/mnist.so"
    param_path = "models/aarch64/mnist.params"
    module_lib.export_library(lib_path, tvm.contrib.cc.cross_compiler('aarch64-linux-gnu-g++'))
elif target_type == 'opencl':
    lib_path   = "models/opencl/mnist.so"
    param_path = "models/opencl/mnist.params"
    module_lib.export_library(lib_path, tvm.contrib.cc.cross_compiler('aarch64-linux-gnu-g++'))
"""

with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))
