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
#device_key = "v9h" # rename as your device chip type 
device_key = "v9h_m" # rename as your device chip type 
#device_key = "v9t" # rename as your device chip type 
rpc_host = "192.168.1.18" # replace your pc host ip
rpc_port = 9190 # replace the tracker server port
print("target device:", device_key)
print("rpc_host: %s:%s" % (rpc_host, rpc_port))

core_type = -1
core_num  = 6
# Define the neural network and compilation target.
batch_size = 1
layout = "NCHW"

tune_trials = 1200
tune_enable = True 
tune_enable = False 
use_ndk = False

## model select

#network_name = "model-01"
#network_name = "model-01-half"

#network_name = "model-01-half-right"
#network_name = "model-01-half-left"

network_name = "mobilenet"

if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
elif target_type == 'opencl':
    #target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
    target = tvm.target.Target("opencl -device=mali -max_num_threads=512", host="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
print("targte device:", target)


pretrained_model_path = "../pretrained_models"
auto_schedule_log_path = "../AutoSchedule_log_files"
model_path = "%s/%s" % (auto_schedule_log_path, network_name)
# set the name of log file
current_log_file = "%s-%s-C%s-T%s.json" % (network_name, target_type, tune_trials, time.strftime('%y-%m-%d-%H-%M',time.localtime(time.time())))
history_log_file = "%s-%s-history-log.json" % (network_name, target_type)
best_log_file = "%s-%s-history-log.json.best.json" % (network_name, target_type)
record_log_file = "%s-%s-record.txt" % (network_name, target_type)
latency_file =  "%s_%s_total_latency.tsv" % (network_name, target_type)

current_log_path = os.path.join(model_path, current_log_file)
history_log_path = os.path.join(model_path, history_log_file)
best_log_path = os.path.join(model_path, best_log_file)
record_log_path = os.path.join(model_path, record_log_file)
latency_path = os.path.join(model_path, latency_file)
print("current log path:", current_log_path)
print("history log path:", history_log_path)
print("record log path:", record_log_path)
print("latency path:", latency_path)

if not os.path.exists(auto_schedule_log_path):
    os.system("mkdir -p %s" % (auto_schedule_log_path))
if not os.path.exists(model_path):
    os.system("mkdir -p %s" % (model_path))

if network_name == "model-01" or network_name == "model-01-warp32" or network_name == "model-01-mali":
    # configure the model net
    blob_file  = os.path.join(pretrained_model_path, "model-01/model_best_20210826.caffemodel")
    proto_file = os.path.join(pretrained_model_path, "model-01/model_best_20210826.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (1, 3, 424, 336);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    if False:
        # convert the model to ir module
        model, params = relay.frontend.from_caffe2(blob_file, proto_file, shape_dict, dtype_dict)
    else:
        from google.protobuf import text_format
        #from tvm.relay.frontend import caffe_pb2 as pb
        import caffe
        from caffe import layers as L, params as P
        from caffe.proto import caffe_pb2 as pb
        init_net = pb.NetParameter()
        predict_net = pb.NetParameter()

        # load model
        with open(proto_file, "r") as f:
            text_format.Merge(f.read(), predict_net)
        # load blob
        with open(blob_file, "rb") as f:
            init_net.ParseFromString(f.read())

        model, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)
elif network_name == "model-01-half" or network_name == "model-01-half-mali":
    # configure the model net
    blob_file  = os.path.join(pretrained_model_path, "model-01-half/small_6303_0_1lr11_iter_80000.caffemodel")
    proto_file = os.path.join(pretrained_model_path, "model-01-half/deploy_halfsize.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (1, 3, 640, 480);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    if False:
        # convert the model to ir module
        model, params = relay.frontend.from_caffe2(blob_file, proto_file, shape_dict, dtype_dict)
    else:
        from google.protobuf import text_format
        import caffe
        from caffe import layers as L, params as P
        from caffe.proto import caffe_pb2 as pb
        init_net = pb.NetParameter()
        predict_net = pb.NetParameter()

        # load model
        with open(proto_file, "r") as f:
            text_format.Merge(f.read(), predict_net)
        # load blob
        with open(blob_file, "rb") as f:
            init_net.ParseFromString(f.read())

        model, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)
elif network_name == "model-01-half-right" or network_name == "model-01-half-right-warp32":
    onnx_file  = os.path.join(pretrained_model_path, "model-01-half-right/model_best_20210826_half_right.onnx")
    onnx_model = onnx.load(onnx_file)

    input_name = "input.1"
    input_dtype = "float32"
    input_shape = (1, 3, 424, 336)
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}

    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "model-01-half-left":
    onnx_file  = os.path.join(pretrained_model_path, "model-01-half-left/model_best_20210826_half_left.onnx")
    onnx_model = onnx.load(onnx_file)

    input_name = "input.1"
    input_dtype = "float32"
    input_shape = (1, 3, 424, 336)
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}

    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "mnist":
    mnist_model = onnx.load('mnist/mnist-8.onnx')

    input_name = "Input3"
    dtype = "float32"
    input_shape = (1, 1, 28, 28)
    shape_dict = {input_name: input_shape}
    model, params = relay.frontend.from_onnx(mnist_model, shape_dict)
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

if tune_enable:
    if target_type == 'opencl':
        num_cores = 2
        vector_unit_bytes = 16
        cache_line_bytes  = 64
        max_shared_memory_per_block = 4096
        max_local_memory_per_block  = 2147483647 #40960
        max_threads_per_block = 512
        max_vthread_extent = 1
        warp_size = 1
        
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
        # configure the tune option parameter for opencl
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=tune_trials,  # change this to 20000 to achieve the best performance
            builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
            runner=auto_scheduler.RPCRunner(device_key, host=rpc_host, port=rpc_port, 
                                            number=5, repeat=3, timeout=30, min_repeat_ms=150),
            measure_callbacks=[auto_scheduler.RecordToFile(current_log_path)],
        )
    else:
        tasks, task_weights = auto_scheduler.extract_tasks(model["main"], params, target)
        # configure the tune option parameter for opencl
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=tune_trials,  # change this to 20000 to achieve the best performance
            builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
            runner=auto_scheduler.RPCRunner(device_key, host=rpc_host, port=rpc_port, 
                                            number=5, repeat=3, timeout=15, min_repeat_ms=150,
                                            enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(current_log_path)],
        )

    ## print the extrack task
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
    for index, task_weight in enumerate(task_weights):
        print("========== Task %d  (weight: %s) ==========" % (index, task_weight))

    ## configure the schedule callback
    scheduler_callbacks = [
        auto_scheduler.task_scheduler.PrintTableInfo(),
        auto_scheduler.task_scheduler.LogEstimatedLatency(latency_path),
    ]
    if os.path.exists(history_log_path):
        print("Loaded pretuning log file:", history_log_path)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=history_log_path, callbacks=scheduler_callbacks)
    else:
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, callbacks=scheduler_callbacks)

    print("=============== Begin tuning... ===============")
    ## save the log file name 
    with open(record_log_path, 'a+') as t_f:
        t_f.write(current_log_file + '\n')
    ## begin tune
    tuner.tune(tune_option)
    # update history log file append the current tune log file
    os.system("cat %s >> %s" % (current_log_path, history_log_path))
else:
    # Compile the whole network
    print("=============== Compile...  ===============")
    # if the turn is disabled, can load the exist log file
    if os.path.exists(best_log_path):
        print("Compile Loaded File:", best_log_path)
        with auto_scheduler.ApplyHistoryBest(best_log_path):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(model, target, params=params)
    else:
        #history_log_path = "model_best_20210826_half_right-NCHW-B1-opencl-C20000-T21-09-06-10-00.json"
        print("Compile Loaded File:", history_log_path)
        with auto_scheduler.ApplyHistoryBest(history_log_path):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(model, target, params=params)
    # Create graph executor
    print("=============== Request Remote ===============")
    if target_type == 'x86_64':
        remote = rpc.LocalSession()
        #remote = request_remote(device_key, rpc_host, rpc_port)
        dev = remote.cpu(0)
    elif target_type == 'aarch64':
        remote = request_remote(device_key, rpc_host, rpc_port)
        dev = remote.cpu()
        # configure the number of cpu running cores
        config_func = remote.get_function('runtime.config_threadpool')
        config_func(core_type, core_num)
    elif target_type == 'opencl':
        remote = request_remote(device_key, rpc_host, rpc_port, timeout=10000)
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
    data = (np.random.uniform(size=input_shape)).astype(input_dtype)
    data_tvm = tvm.nd.array(data)
    module.set_input(input_name, data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    #ftimer = module.module.time_evaluator("run", dev, number=50, repeat=3)
    ftimer = module.module.time_evaluator("run", dev, number=50, repeat=3, min_repeat_ms=100)
    #ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

