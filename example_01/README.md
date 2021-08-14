# Example one
This example introduce one way to use `tvm rpc server` to deploy the **mnist** model  on `x86_64` , `aarch64`or `opencl` device.

## 1. Device
### 1.1. Dependent library
- **TVM Runtime**
1. libtvm-runtime.so

In the `tvm` folder, run `./build.sh deploy` will generate the `tvm_runtime_deploy.run` file in `build_aarch64` folder,  if target device don't contain this dynamic library, you should copy this file to target device and execute command as follow:

```shell
$ chmod 777 tvm_runtime_deploy.run
$ ./tvm_runtime_deploy.run
```

This will auto deploy the `libtvm-runtime.so` to `/usr/sdrv/tvm` folder.

### 1.2. Python library

In device side,check whether installed `psutil` and `cloudpickle` package.

```shell
$ python3 -m pip list | grep psutil
$ python3 -m pip list | grep cloudpickle
```

If these package exist, then ignore next step, otherwise install these package.

```shell
$ python3 -m pip install psutil cloudpickle
```

### 1.3.  Device IP 

Query target device ip address and make sure the device and host to be in the same network segment.

```shell
$ ifconfig
```

### 1.4 RPC Server

Establish the **RPC** server on target device.

```shell
$ python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090
```

if rpc server is successfully established, it will show message as follow:

```
INFO:RPCServer:bind to 0.0.0.0:9090
```

## 2. Host

1. Revise the target device ip address.

Open the `/python/rpc_deploy.py` file and replace  `host="192.168.104.240"` to your device actual `ip` address.

2. running `build.sh` script for model build.

```shell
$ ./build.sh
```

3. select the target device type

```bash
$ ./build.sh
---| number | target type|
-->   [1]   |   x86_64   |
-->   [2]   |   aarch64  |
-->   [3]   |   opencl   |
-->   [4]   |   all      |
Enter target type number: 1
```
If the enter number is `1` ,the script will deploy model on `x86_64` device.

### 2.3 Result

```shell
---| number | target type|                                                                                                       
-->   [1]   |   x86_64   | 
-->   [2]   |   aarch64  |     
-->   [3]   |   opencl   |         
-->   [4]   |   all      |
Enter target type number: 4                                                                                                      
all build      
device: llvm -keys=cpu -link-params=0
host: llvm -keys=cpu -link-params=0
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7fa904bf9390>
Target type: x86_64
TVM prediction number:  8
Evaluate inference time cost...
Mean inference time (std dev): 0.08 ms (0.02 ms)
device: llvm -keys=arm_cpu,cpu -device=arm_cpu -link-params=0 -mattr=+neon -mtriple=aarch64-linux-gnu
host: llvm -keys=arm_cpu,cpu -device=arm_cpu -link-params=0 -mattr=+neon -mtriple=aarch64-linux-gnu
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7f2ea118b358>
Target type: aarch64
TVM prediction number:  8
Evaluate inference time cost...
Mean inference time (std dev): 0.23 ms (0.08 ms)
device: opencl -keys=mali,opencl,gpu -device=mali -max_num_threads=256 -thread_warp_size=1
host: llvm -keys=arm_cpu,cpu -device=arm_cpu -link-params=0 -mattr=+neon -mtriple=aarch64-linux-gnu
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
[16:07:09] /home/zgh/Workspace/github/tvm/src/runtime/opencl/opencl_device_api.cc:400: Warning: Using CPU OpenCL device
<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7ff2e71d8358>
Target type: opencl
TVM prediction number:  5
Evaluate inference time cost...
Mean inference time (std dev): 6.09 ms (0.79 ms)
```

