#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import tvm.contrib.graph_executor as runtime


# In[2]:


# Also replace this with the device key in your tracker
device_key = "rasp"
if device_key == "v9h":
    rpc_host = "192.168.105.70"
elif device_key == "rasp":
    rpc_host = "192.168.0.109"
rpc_port = 9190
print("device:", device_key)
print("rpc_host: %s:%s" % (rpc_host, rpc_port))


# In[3]:


# Define the neural network and compilation target.

# network = "mobilenet"
network = "mnist"
batch_size = 1
layout = "NCHW"
dtype = "float32"
# device_type = "armv7"
# device_type = "aarch64"
device_type = "opencl"


# In[4]:


turn_trials = 200
turn_enable = True
preload_log_file = False
# Set this to True if you use ndk tools for cross compiling
use_ndk = False
# Path to cross compiler
# os.environ["TVM_NDK_CC"] = "/usr/bin/aarch64-linux-gnu-g++"


# In[5]:


if network == "mobilenet":
    tune_model = onnx.load('../mobilenet/mobilenetv2-7.onnx')
    input_name = "input"
    input_shape = (batch_size, 3, 244, 244)
    shape_dict = {input_name: input_shape}
    print("shape_dict: ", shape_dict)
elif network == "mnist":
    tune_model = onnx.load('../mnist/mnist-8.onnx')
    input_name = "Input3"
    input_shape = (batch_size, 1, 28, 28)
    shape_dict = {input_name: input_shape}
    print("shape_dict: ", shape_dict)


# In[6]:


model, params = relay.frontend.from_onnx(tune_model, shape_dict)


# In[7]:


if device_type == "armv7":
    target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf")
elif device_type == "aarch64":
    target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
elif device_type == "opencl":
#     target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")
    target = tvm.target.Target("opencl -device=powervr -model=v9h", host="llvm -mtriple=aarch64-linux-gnu")

log_file = "%s-%s-%s-B%d-%s-C%s-T%s.log" % (device_key, network, layout, batch_size, target.kind.name, turn_trials, time.strftime('%y-%m-%d-%H-%M',time.localtime(time.time())))
print("log file:", log_file)


# In[8]:


if layout == 'NHWC':
    # convert from NCHW to NHWC
    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}

    # Convert the layout to NHWC
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])

    with tvm.transform.PassContext(opt_level=3):
        model = seq(model)


# In[9]:


# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": turn_trials,
#     "early_stopping": 800,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
#         runner=autotvm.RPCRunner(
#             key=device_key,
#             host=rpc_host,
#             port=rpc_port,
#             number=5,
#             timeout=10,
#             min_repeat_ms = 200,
#             enable_cpu_cache_flush=True
#         ),
#     ),
# }
use_transfer_learning = True
log_filename = log_file
tuner = "xgb"
n_trial = turn_trials
early_stopping = 800
measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="ndk" if use_ndk else "default"),
                    runner=autotvm.RPCRunner(
                        key=device_key,
                        host=rpc_host,
                        port=rpc_port,
                        number=5,
                        timeout=10,
                        min_repeat_ms = 200,
                        enable_cpu_cache_flush=True)
                )


# In[10]:


# extract workloads from relay program
print("Extract tasks...")
tasks = autotvm.task.extract_from_program(
    model["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
)

for idx, task in enumerate(tasks):
    print("========== Task %d  (function name: %s) ==========" % (idx, task.name))
    print(task.args)


# In[11]:


# create tmp log file
tmp_log_file = log_filename + ".tmp"
if os.path.exists(tmp_log_file):
    os.remove(tmp_log_file)


# In[12]:


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
        if os.path.isfile(tmp_log_file):
            tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

    # process tuning
    tsk_trial = min(n_trial, len(tsk.config_space))
    tuner_obj.tune(
        n_trial=tsk_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
            autotvm.callback.log_to_file(tmp_log_file),
        ],
    )


# In[ ]:


# pick best records to a cache file
autotvm.record.pick_best(tmp_log_file, log_filename)
os.remove(tmp_log_file)


# In[ ]:


# compile kernels with history best records
with autotvm.apply_history_best(log_file):
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(model, target=target, params=params)


# In[ ]:


#  export library
temp = utils.tempdir()
filename = device_type + "_deploy_lib.tar"
path_lib = temp.relpath(filename)
lib.export_library(path_lib)
# lib.export_library(path_lib, ndk.create_shared)


# In[ ]:


print("=============== Request Remote ===============")
remote = request_remote(device_key, rpc_host, rpc_port)
# upload module to device
remote.upload(path_lib)
loaded_lib = remote.load_module(filename)
if device_type == "armv7":
    dev = remote.cpu()
elif device_type == "aarch64":
    dev = remote.cpu()
elif device_type == "opencl":
    dev = remote.cl()


# In[ ]:


# Create graph executor
module = runtime.GraphModule(loaded_lib["default"](dev))
# module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype)))


# In[ ]:


# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=50)


# In[ ]:


prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


# In[ ]:




