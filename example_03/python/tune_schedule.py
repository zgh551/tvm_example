#!/usr/bin/env python
# coding: utf-8

import onnx
import glob
import os
import numpy as np
import time
import sys
import tvm
from tvm import rpc
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
from tvm.contrib import utils, ndk
from tvm.auto_scheduler.utils import request_remote

# get the target device type 
target_type = sys.argv[1]

# Also replace this with the device key in your tracker
device_key = "v9h" # rename as your device chip type 
rpc_host = "192.168.0.109" # replace your pc host ip
rpc_port = 9190 # replace the tracker server port
print("target device:", device_key)
print("rpc_host: %s:%s" % (rpc_host, rpc_port))

mnist_model = onnx.load('mnist/mnist-8.onnx')

input_name = "Input3"
dtype = "float32"
input_shape = (1, 1, 28, 28)
shape_dict = {input_name: input_shape}
print("shape_dict: ", shape_dict)

model, params = relay.frontend.from_onnx(mnist_model, shape_dict)

# Define the neural network and compilation target.
network = "mnist"
batch_size = 1
layout = "NCHW"

turn_trials = 200
turn_enable = True
preload_log_file = False
use_ndk = False

if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
elif target_type == 'opencl':
    target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")
print("device:", target)

# set the name of log file
log_file = "%s-%s-B%d-%s-C%s-T%s.json" % (network, layout, batch_size, target.kind.name, turn_trials, time.strftime('%y-%m-%d-%H-%M',time.localtime(time.time())))
print("log file:", log_file)

if turn_enable:
    if target_type == 'opencl':
        num_cores = 2
        vector_unit_bytes = 16
        cache_line_bytes  = 64
        max_shared_memory_per_block = 4096
        max_local_memory_per_block  = 40960
        max_threads_per_block = 512
        max_vthread_extent = 2
        warp_size = 2
        
        print("=============== Hardware Paramter: ===============")
        print("number of cores:", num_cores)
        print("vector unit bytes:", vector_unit_bytes)
        print("cache line bytes:", cache_line_bytes)
        print("max_shared_memory_per_block:", max_shared_memory_per_block)
        print("max_local_memory_per_block:", max_local_memory_per_block)
        print("max_threads_per_block:", max_threads_per_block)
        print("max_vthread_extent:", max_vthread_extent)
        print("warp_size: ", warp_size)
        
        # configure the scheduler parameters for device
        hardware_params = auto_scheduler.HardwareParams(num_cores, vector_unit_bytes, cache_line_bytes,
                                                        max_shared_memory_per_block, max_local_memory_per_block,
                                                        max_threads_per_block, max_vthread_extent, warp_size)
        
        # extrack the task from model
        tasks, task_weights = auto_scheduler.extract_tasks(model["main"], params, target, hardware_params=hardware_params)
    else:
        tasks, task_weights = auto_scheduler.extract_tasks(model["main"], params, target)
        
    print("=============== Begin tuning... ===============")
    if preload_log_file:
        # if have the log file for scheduler, repalce follow name of file:
        load_log_file = "xxx.json"
        print("preload file:", load_log_file)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=load_log_file)
    else:
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

    # configure the tune option parameter
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=turn_trials,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=auto_scheduler.RPCRunner(device_key, host=rpc_host, port=rpc_port, repeat=3, timeout=50),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    
    # begin tune
    tuner.tune(tune_option)

# Compile the whole network
print("=============== Compile...  ===============")
# if the turn is disabled, can load the exist log file
if not turn_enable:
    log_file = "xxx.json"
print("Load File:", log_file)
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(model, target, params=params)

# Create graph executor
print("=============== Request Remote ===============")
if target_type == 'x86_64':
    #remote = rpc.LocalSession()
    remote = request_remote(device_key, rpc_host, rpc_port)
    dev = remote.cpu(0)
elif target_type == 'aarch64':
    remote = request_remote(device_key, rpc_host, rpc_port)
    dev = remote.cpu()
elif target_type == 'opencl':
    remote = request_remote(device_key, rpc_host, rpc_port)
    dev = remote.cl()

# load model to remote device and set random input data
print("=============== Load Model ===============")
temp = utils.tempdir()
filename = "deploy_lib.tar"
path_lib = temp.relpath(filename)
lib.export_library(path_lib)
remote.upload(path_lib)
loaded_lib = remote.load_module(filename)
module = graph_executor.GraphModule(loaded_lib["default"](dev))
data = (np.random.uniform(size=input_shape)).astype(dtype)
data_tvm = tvm.nd.array(data)
module.set_input(input_name, data_tvm)

# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

