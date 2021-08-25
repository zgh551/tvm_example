#!/usr/bin/env python
# coding: utf-8
import onnx
import glob
import os
import numpy as np
import time

import tvm
from tvm.auto_scheduler.utils import request_remote
from tvm.contrib import utils, ndk
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
import logging
import sys

# configure whether enable tune
turn_enable = False  
use_transfer_learning = True 
# configure the tune parameters
turn_trials = 1000
early_stopping = 800
tuner = "xgb"
# Set this to True if you use ndk tools for cross compiling
use_ndk = False
# Also replace this with the device key in your tracker
device_key = "v9h"
# repalce the host ip of your pc 
rpc_host = "192.168.1.18"
rpc_port = 9190
print("device:", device_key)
print("rpc_host: %s:%s" % (rpc_host, rpc_port))
# configure the number of running core
core_type = -1
core_num  = 2
# get the target device type 
target_type  = sys.argv[1]
network_name = sys.argv[2]
# configure the layout parameters
batch_size = 1
layout = "NCHW"
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
    model_dir = "op9_dla"
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

if target_type == "x86_64":
    target = tvm.target.Target("llvm")
elif target_type == "armv7":
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=armv7l-linux-gnueabihf -mattr=+neon")
elif target_type == "aarch64":
    target = tvm.target.Target("llvm -device=arm_cpu -mcpu=cortex-a55 -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod")
elif target_type == "opencl":
    target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")

log_file_path = "auto_tvm_log/"
temp_log_filename = "%s-%s-%s-%s-C%s-T%s.log" % (device_key, network_name, layout, target_type, turn_trials, 
                                            time.strftime('%y-%m-%d-%H-%M',time.localtime(time.time())))
log_filename = "%s-%s-%s-%s.log" % (device_key, network_name, layout, target_type)
# create tmp log file name and log file name
tmp_log_file = log_file_path + temp_log_filename + ".tmp"
log_file     = log_file_path + log_filename
print("temp log file: name:", temp_log_filename)
print("log file: name:", log_filename)


if layout == 'NHWC':
    # convert from NCHW to NHWC
    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
    # Convert the layout to NHWC
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        model = seq(model)

if turn_enable:
    if target_type == "x86_64":
        #Set number of threads used for tuning based on the number of
        # physical CPU cores on your machine.
        num_threads = 8
        os.environ["TVM_NUM_THREADS"] = str(num_threads)

        measure_option = autotvm.measure_option(
            builder = autotvm.LocalBuilder(),
            runner  = autotvm.LocalRunner(
                number=1,
                repeat=10,
                min_repeat_ms = 0,
                enable_cpu_cache_flush=True
            ),
        )
    elif target_type == "armv7" or target_type == "aarch64":
        measure_option = autotvm.measure_option(
            builder = autotvm.LocalBuilder(build_func="ndk" if use_ndk else "default"),
            runner  = autotvm.RPCRunner(
                key = device_key,
                host= rpc_host,
                port= rpc_port,
                number = 5,
                timeout= 10,
                enable_cpu_cache_flush = True
            ),
        )
    elif target_type == "opencl":
        measure_option = autotvm.measure_option(
            builder = autotvm.LocalBuilder(build_func="ndk" if use_ndk else "default"),
            runner  = autotvm.RPCRunner(
                key = device_key,
                host= rpc_host,
                port= rpc_port,
                repeat = 3,
                number = 5,
                timeout= 10,
                min_repeat_ms = 150,
            ),
        )

    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        model["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"))
    )

    for idx, task in enumerate(tasks):
        print("========== Task %d (function name: %s) [%f GFLOPS] ==========" % (idx, task.name, task.flop * 1.0e-6))
        print(task.args)

    # run tuning tasks
    print("Tuning...")
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        
        if use_transfer_learning:
            if os.path.isfile(log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(log_file))

        # process tuning
        tsk_trial = min(turn_trials, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )


    if target_type == "x86_64":
        # Use graph tuner to achieve graph level optimal schedules
        # Set use_DP=False if it takes too long to finish.
        use_DP = True
        target_op = [relay.op.get("nn.conv2d"),]
        Tuner = DPTuner if use_DP else PBQPTuner
        executor = Tuner(model["main"], shape_dict, tmp_log_file, target_op, target)
        executor.benchmark_layout_transform(min_exec_num=2000)
        executor.run()
        executor.write_opt_sch2record_file(log_file)
    else:
        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_file)
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
else:
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        model["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"))
    )

    #print("task configure space:", tasks.config_space)
    for idx, task in enumerate(tasks):
        print("========== Task %d (function name: %s) [%f GFLOPS] ==========" % (idx, task.name, task.flop * 1.0e-6))
        #print("task configure space:", task.config_space)
        print(task.args)


# compile kernels with history best records
with autotvm.apply_history_best(log_file):
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(model, target=target, params=params)

    print(lib)
    dev_module = lib.get_lib()
    print(dev_module)
    #source_code = dev_module.imported_modules()

    if target_type == "x86_64":
        # upload parameters to device
        dev = tvm.cpu()
        module = runtime.GraphModule(lib["default"](dev))
    else:
        #  export library
        print("Export library...")
        temp = utils.tempdir()
        exp_lib_name = target_type + "_deploy_lib.tar"
        path_lib = temp.relpath(exp_lib_name)
        lib.export_library(path_lib)
        print("Lib:", exp_lib_name)

        print("=============== Request Remote ===============")
        remote = request_remote(device_key, rpc_host, rpc_port)

        # upload module to device
        print("Load Module...")
        remote.upload(path_lib)
        loaded_lib = remote.load_module(exp_lib_name)
        # configure the device
        if target_type == "armv7":
            dev = remote.cpu()
        elif target_type == "aarch64":
            dev = remote.cpu()
            config_func = remote.get_function('runtime.config_threadpool')
            config_func(core_type, core_num)
        elif target_type == "opencl":
            dev = remote.cl()
        # Create graph executor
        module = runtime.GraphModule(loaded_lib["default"](dev))

    module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype)))
    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=50)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

