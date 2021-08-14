# Example One
This example introduce one way to use `tvm rpc server` to deploy the **mnist** model  on `x86_64` , `arm`or `gpu` device.

> Note: rpc server current only supports ubuntu but not yocto os.

## 1. Device
### 1.1. Dependent Library
- **TVM Runtime**
1. libtvm-runtime.so

In the `tvm` folder, run `./build.sh deploy` will generate the `tvm_runtime_deploy.run` file in `build_aarch64` folder,  if target device don't contain this dynamic library, you should copy this file to target device and execute command as follow:

```shell
$ chmod 777 tvm_runtime_deploy.run
$ ./tvm_runtime_deploy.run
```

This will auto deploy the `libtvm-runtime.so` to `/usr/sdrv/tvm` folder.

### 1.2. Python Library

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

### 2.1. Revise Target Device `IP`

Open the `{example_base_dir}/python/rpc_deploy.py` file and replace  `host="192.168.104.240"` to your device actual `ip` address.

### 2.2. Running Script

1. running `build.sh` script for model build.

```shell
$ ./build.sh
```

2. select the target device type

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
Enter target type number: 1                                                                                                      
x86_64 build      
device: llvm -keys=cpu -link-params=0
host: llvm -keys=cpu -link-params=0
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7fa904bf9390>
Target type: x86_64
TVM prediction number:  8
Evaluate inference time cost...
Mean inference time (std dev): 0.08 ms (0.02 ms)
```

