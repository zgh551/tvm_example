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
from tvm import relay, auto_scheduler


import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata

from tvm.auto_scheduler.utils import request_remote

#logging.getLogger("autotvm").setLevel(logging.INFO)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

os.environ["TVM_BACKTRACE"] = str(1)
# The following is my environment, change this to the IP address of
# your target device
host = "192.168.105.54"
#host = "192.168.0.164"
port = 9090

"""
device_key = "v9h_m" # rename as your device chip type 
rpc_host = "192.168.1.18" # replace your pc host ip
rpc_port = 9190 # replace the tracker server port
print("target device:", device_key)
print("rpc_host: %s:%s" % (rpc_host, rpc_port))
"""

# selecrt whether model quantify
quantify_enable = False
auto_tune_build = True
auto_tune_build = False 
# configure the cpu cores
core_type = -1
core_num  = 6

## batch size
batch_size = 1
# get the target device type 
target_type  = sys.argv[1]
network_name = sys.argv[2]
#network_name = "jreg_caffe"
#network_name = "jreg_onnx"
#network_name = "jseg_caffe"
#network_name = "yfast_onnx"
#network_name = "model-01"
#network_name = "model-02"
#network_name = "model-03"
#network_name = "mobilenet"
#network_name = "resnet"

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

elif network_name == "op9_dla_onnx":
    model_path = os.path.join(pretrained_model_path, "op9_dla/onnx/op9_dla.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input.1"
    input_shape = (batch_size, 320, 40, 24);
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
    input_shape = (batch_size, 320, 40, 24);
    input_dtype = "int8"
    #input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_tflite(tflite_model, shape_dict, dtype_dict)

elif network_name == "jreg_caffe":
    # configure the model net
    blob_file = os.path.join(pretrained_model_path, "jreg_288hx240w/caffemodel/jreg.caffemodel")
    #proto_file = os.path.join(pretrained_model_path, "jreg_288hx240w/caffemodel/jreg_deploy_revise3.prototxt")
    proto_file = os.path.join(pretrained_model_path, "jreg_288hx240w/caffemodel/jreg_deploy.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (batch_size, 3, 288, 240);
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
        #print(model)
elif network_name == "jreg_onnx":
    model_path = os.path.join(pretrained_model_path, "jreg_288hx240w/onnx/JSeg_optset11.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input.1"
    input_shape = (batch_size, 3, 288, 240);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    #onnx_model = relay.transform.DynamicToStatic()(onnx_model)
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "jseg_caffe":
    # configure the model net
    blob_file = os.path.join(pretrained_model_path, "jseg_224hx256w/jseg.caffemodel")
    proto_file = os.path.join(pretrained_model_path, "jseg_224hx256w/deploy_256x224.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (batch_size, 3, 224, 256);
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
elif network_name == "yfast_onnx":
    model_path = os.path.join(pretrained_model_path, "yfast_128hx128w/yfast_128x128.onnx")
    onnx_model = onnx.load(model_path)
    # configure the model input ,shape ans type
    input_name = "input"
    input_shape = (batch_size, 3, 128, 128);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    # convert the model to ir module
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
elif network_name == "model-01":
    # configure the model net
    blob_file  = os.path.join(pretrained_model_path, "model-01/model_best_20210826.caffemodel")
    proto_file = os.path.join(pretrained_model_path, "model-01/model_best_20210826.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (batch_size, 3, 424, 336);
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
elif network_name == "model-02":
    # configure the model net
    blob_file  = os.path.join(pretrained_model_path, "model-02/20200411_bgr_56_iter_40000.caffemodel")
    proto_file = os.path.join(pretrained_model_path, "model-02/deploy.prototxt")
    # configure the model input ,shape ans type
    input_name = "data"
    input_shape = (batch_size, 3, 56, 56);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    if False:
        # convert the model to ir module
        model, params = relay.frontend.from_caffe2(blob_file, proto_file, shape_dict, dtype_dict)
    else:
        init_net = pb.NetParameter()
        predict_net = pb.NetParameter()

        # load model
        with open(proto_file, "r") as f:
            text_format.Merge(f.read(), predict_net)
        # load blob
        with open(blob_file, "rb") as f:
            init_net.ParseFromString(f.read())

        model, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)
elif network_name == "model-03":
    # configure the model net
    blob_file  = os.path.join(pretrained_model_path, "model-03/20210828_model.caffemodel")
    proto_file = os.path.join(pretrained_model_path, "model-03/20210828_model.prototxt")
    # configure the model input ,shape ans type
    input_name = "blob1"
    input_shape = (batch_size, 3, 192, 320);
    input_dtype = "float32"
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    if False:
        # convert the model to ir module
        model, params = relay.frontend.from_caffe2(blob_file, proto_file, shape_dict, dtype_dict)
    else:
        init_net = pb.NetParameter()
        predict_net = pb.NetParameter()

        # load model
        with open(proto_file, "r") as f:
            text_format.Merge(f.read(), predict_net)
        # load blob
        with open(blob_file, "rb") as f:
            init_net.ParseFromString(f.read())

        model, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)



if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target      = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
elif target_type == 'opencl':
    #target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
    target = tvm.target.Target("opencl -max_num_threads=512", host="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
    #target = tvm.target.Target("opencl -device=mali -max_num_threads=512", host="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
    #target = tvm.target.Target("opencl -max_num_threads=256", host="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a55 -mattr=+neon,+v8.2a,+dotprod")
print("device:", target)

if os.path.exists(best_log_path):
    print("Compile Loaded File:", best_log_path)
    with auto_scheduler.ApplyHistoryBest(best_log_path):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            model_lib = relay.build(model, target, params=params)
elif os.path.exists(history_log_path):
    print("Compile Loaded File:", history_log_path)
    with auto_scheduler.ApplyHistoryBest(history_log_path):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            model_lib = relay.build(model, target, params=params)
else: 
    with tvm.transform.PassContext(opt_level=3):
        model_lib = relay.build(model, target=target, params=params)
print(model_lib)
print(model_lib.lib.imported_modules)
print(model_lib.target)

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
model_lib.export_library(lib_fname)
print("export library")

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
    #remote = request_remote(device_key, rpc_host, rpc_port, timeout=10000)
    print("connect finish")
    dev = remote.cl()
print("get remote device")

# upload the library to remote device and load it
remote.upload(lib_fname)
print("loading module")
rlib = remote.load_module("net.tar")
module = runtime.GraphModule(rlib["default"](dev))

if network_name == "mobilenet" or network_name == "resnet":
    if False:
        import mxnet as mx
        import gluoncv
        from mxnet import gluon, nd
        from mxnet.gluon.data.vision import transforms
        from gluoncv.data import imagenet
        #img_path = '/home/guohua.zhu/Workspace/github/tvm_example/example_01/img_dataset/'
        #img_path = '/workspace/guohua.zhu/img_dataset/'
        #img_path = '/workspace/guohua.zhu/imagenet/'
        img_path = '/home/guohua.zhu/workspace/guohua.zhu/imagenet/'
        # batch size (set to 1 for cpu)
        # Define evaluation metrics
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)

        # Define image transforms
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if True:
            # Load and process input
            val_data = gluon.data.DataLoader(
                imagenet.classification.ImageNet(img_path, train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False)

            # Compute evaluations
            #Batch = namedtuple('Batch', ['data'])
            acc_top1.reset()
            acc_top5.reset()

            num_batches = int(50000/batch_size)
            print('[0 / %d] batches done'%(num_batches))
            # Loop over batches
            for i, batch in enumerate(val_data):
                # Load batch
                #data = tvm.nd.array(gluon.utils.split_and_load(batch[0], batch_axis=0)
                #label = gluon.utils.split_and_load(batch[1], batch_axis=0)
                # Perform forward pass
                data = batch[0].asnumpy()
                label = batch[1]
                module.set_input(input_name, tvm.nd.array(data.astype(input_dtype))) 
                #module.set_input(input_name, tvm.nd.array(image_data.astype(input_dtype))) 
                module.run()
                tvm_output = mx.nd.array(module.get_output(0).numpy())

                # Update accuracy metrics
                acc_top1.update(label, tvm_output)
                acc_top5.update(label, tvm_output)

                if (i+1)%50==0:
                    print('[%d / %d] batches done'%(i+1,num_batches))
                    # Print results
                    _, top1 = acc_top1.get()
                    _, top5 = acc_top5.get()
                    print("Top-1 accuracy: {}, Top-5 accuracy: {}".format(top1, top5))
        else:
            val_dataset = gluoncv.data.ImageNet(img_path, train=False).transform_first(transform_test)

            data = val_dataset[1234][0].asnumpy()
            label = val_dataset[1234][1]

            module.set_input(input_name, tvm.nd.array(data.astype(input_dtype))) 
            module.run()
            tvm_output = module.get_output(0).numpy()

            synset_url = "".join(
                [
                    "https://gist.githubusercontent.com/zhreshold/",
                    "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                    "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                    "imagenet1000_clsid_to_human.txt",
                ]
            )
            synset_name = "imagenet1000_clsid_to_human.txt"
            #img_path = download_testdata(img_url, "cat.png", module="data")
            synset_path = download_testdata(synset_url, synset_name, module="data")

            with open(synset_path) as f:
                synset = eval(f.read())

            top1 = np.argmax(tvm_output[0])
            print("mobilenet top1:", top1, synset[top1])

            predictions = np.squeeze(tvm_output)
            top_k = predictions.argsort()[-5:][::-1]
            print("mobilenet top5:")
            for node_id in top_k:
                score = predictions[node_id]
                print("index = %d score = %.5f class = %s" % (node_id, score, synset[node_id]))

            """
            # Update accuracy metrics
            tvm_output = mx.nd.array(tvm_output)
            acc_top1.update(label, tvm_output)
            acc_top5.update(label, tvm_output)

            # Print results
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            print("Top-1 accuracy: {}, Top-5 accuracy: {}".format(top1, top5))
            """
    else:
        img_path = "data/cat.png"
        img = Image.open(img_path).resize((224, 224))
        print(img)
        def transform_image(image):
            image = np.array(image) - np.array([123.0, 117.0, 104.0])
            image /= np.array([58.395, 57.12, 57.375])
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis, :]
            return image
        image_data = transform_image(img)
        print("input", image_data.shape)
        print(image_data)
        synset_url = "".join(
            [
                "https://gist.githubusercontent.com/zhreshold/",
                "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                "imagenet1000_clsid_to_human.txt",
            ]
        )
        synset_name = "imagenet1000_clsid_to_human.txt"
        #img_path = download_testdata(img_url, "cat.png", module="data")
        synset_path = download_testdata(synset_url, synset_name, module="data")
        with open(synset_path) as f:
            synset = eval(f.read())
        module.set_input(input_name, tvm.nd.array(image_data.astype(input_dtype))) 
        module.run()
        tvm_output = module.get_output(0).numpy()
        top1 = np.argmax(tvm_output[0])
        print("mobilenet top1:", top1, synset[top1])

        predictions = np.squeeze(tvm_output)

        top_k = predictions.argsort()[-5:][::-1]
        print("mobilenet top5:")
        for node_id in top_k:
            score = predictions[node_id]
            print("index = %d score = %.5f class = %s" % (node_id, score, synset[node_id]))
elif network_name == "mnist": 
    img_path = "mnist_data/8.png"
    img = Image.open(img_path)

    def transform_image(image):
        image = np.array(image)
        #image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, 2:]
        #image = image[np.newaxis, np.newaxis, :]
        return image
    image_data = transform_image(img)
    # set input data
    module.set_input(input_name, image_data.astype(input_dtype))
    #module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))) 
    # run
    module.run()
    # get the result
    number = np.argmax(module.get_output(0).numpy())
    print("mnist prediction number: ", number)
else:
    # evaluate
    module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))) 
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, number=50, repeat=3)
    # convert to millisecond
    prof_res = np.array(ftimer().results) * 1000
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
