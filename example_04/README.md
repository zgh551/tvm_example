# Example Four
This example introduce one way to use `AutoTVM` to tune the `conv` operator and evaluate the performance of  the model  on `x86_64` , `arm` or `gpu` device.

> Note: rpc tracker server current only supports ubuntu but not yocto os.

## Device

### 1.1. Dependent Library
- **TVM Runtime**
1. libtvm_runtime.so

In the `tvm` folder, execute `./build.sh deploy` will generate the `tvm_runtime_deploy.run` file in the `build_aarch64` folder,  if target device don't contain this dynamic library, you should copy this file to target device and execute command as follow:

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

### 1.3.  Device `IP`  Configure

Query target device `ip` address and make sure the device and host to be in the same network segment.

```shell
$ ifconfig
```

### 1.4 RPC Tracker Server

Establish the **rpc tracker server**  on target device. with two parameter to configure, `tracker`paramter set as host `ip` and `port` ,`key` parameter set as device name.

```shell
$ python3 -m tvm.exec.rpc_server --tracker 192.168.105.70:9190 --key v9h
```

if **rpc tracker server** is successfully established, it will show message as follow:

```
INFO:RPCServer:bind to 0.0.0.0:9090
```

## 2. Host

### 2.1. Tracker Server

First, on the pc host open the tracker server,the running script as follow:

```shell
$ python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

if **rpc tracker server** is successfully established, it will show message as follow:

```shell
INFO:RPCTracker:bind to 0.0.0.0:9190
```

It will moniter the ip on port of 9190 and use follow command to query the device whether connect successfully.

```
python3 -m tvm.exec.query_rpc_tracker --host 0.0.0.0 --port 9190
```
it will show message as follow:
```
Tracker address 0.0.0.0:9190

Server List
----------------------------
server-address  key
----------------------------
192.168.0.109:9090      server:v9h
192.168.0.106:9090      server:rasp
----------------------------

Queue Status
----------------------------
key    total  free  pending
----------------------------
rasp   1      1     0      
v9h    1      1     0      
----------------------------
```

It show two device `v9h` and `rasp` connect to the tracker server.

### 2.2. Revise Host `IP`

Open `{example_base_dir}/python/AutoTVM.py` file and replace  `rpc_host="192.168.1.18"` to the actual `ip` address of  host . `rpc_port=9190` set as the host tracker server port and the `device_key=v9h` set as the device name.

### 2.3. Enable Tune

Open `{example_base_dir}/python/AutoTVM.py` file, if first execute the tune task, then set the `tune_enable` to be `True`.if `auto_tvm_log` folder have the log file of model, then you can set the `tune_enable` to be `Flase`.

### 2.4. CPU Running Cores

Open `{example_base_dir}/python/AutoTVM.py` file, you can set the `core_num` variable to configure the number of `cpu` cores for target device. In the target device, you can also set add the environment variable `export TVM_NUM_THREADS=6` to set the number of `cpu` cores.

### 2.5. Running Script

1. running `build.sh` script for tune model and evaluate the model performance.

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
If the enter number is `1` ,the script will tune model on `x86_64` device.

3. select the tune model

```
---| number | models         |
-->   [1]   | mnist          |
-->   [2]   | mobilenet      |
-->   [3]   | op9_dla_onnx   |
-->   [4]   | op9_dla_tflite |
Enter model number: 1
```

if select the number of one, it will tune the `mnist` model.

### 2.3 Result

```shell
---| number | target type|                                                                                                   
-->   [1]   |   x86_64   |
-->   [2]   |   aarch64  |
-->   [3]   |   opencl   |
-->   [4]   |   all      |
Enter target type number: 1
---| number | models         |
-->   [1]   | mnist          |
-->   [2]   | mobilenet      |
-->   [3]   | op9_dla_onnx   |
-->   [4]   | op9_dla_tflite |
Enter model number: 1
x86_64 build for mnist
device: v9h
rpc_host: 192.168.1.18:9190
temp log file: name: v9h-mnist-NCHW-x86_64-C1000-T21-08-25-07-54.log
log file: name: v9h-mnist-NCHW-x86_64.log
Extract tasks...
========== Task 0 (function name: conv2d_NCHWc.x86) [0.313600 GFLOPS] ==========
(('TENSOR', (1, 1, 32, 32), 'float32'), ('TENSOR', (8, 1, 5, 5), 'float32'), (1, 1), (0, 0, 0, 0), (1, 1), 'NCHW', 'NCHW', 'float32')
========== Task 1 (function name: conv2d_NCHWc.x86) [1.254400 GFLOPS] ==========
(('TENSOR', (1, 8, 18, 18), 'float32'), ('TENSOR', (16, 8, 5, 5), 'float32'), (1, 1), (0, 0, 0, 0), (1, 1), 'NCHW', 'NCHW', 'float32')
========== Task 2 (function name: dense_nopack.x86) [0.005130 GFLOPS] ==========
(('TENSOR', (1, 256), 'float32'), ('TENSOR', (10, 256), 'float32'), None, 'float32')
========== Task 3 (function name: dense_pack.x86) [0.005120 GFLOPS] ==========
(('TENSOR', (1, 256), 'float32'), ('TENSOR', (10, 256), 'float32'), None, 'float32')
Tuning...
[Task  1/ 4]  Current/Best:    2.20/   4.17 GFLOPS | Progress: (48/81) | 11.08 s/usr/local/lib/python3.6/dist-packages/xgboost/training.py:17: UserWarning:
 Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
[Task  1/ 4]  Current/Best:    3.05/   4.56 GFLOPS | Progress: (81/81) | 18.05 s Done.
[Task  2/ 4]  Current/Best:    1.62/   4.74 GFLOPS | Progress: (36/36) | 8.12 s Done.
[Task  3/ 4]  Current/Best:   41.00/ 151.52 GFLOPS | Progress: (240/240) | 45.72 s Done.
[Task  4/ 4]  Current/Best:   18.20/ 103.45 GFLOPS | Progress: (64/64) | 12.52 s Done.
2021-08-25 07:56:15,009 INFO Start to benchmark layout transformation...
2021-08-25 07:56:23,401 INFO Benchmarking layout transformation successful.
2021-08-25 07:56:23,401 INFO Start to run dynamic programming algorithm...
2021-08-25 07:56:23,401 INFO Start forward pass...
2021-08-25 07:56:23,401 INFO Finished forward pass.
2021-08-25 07:56:23,401 INFO Start backward pass...
2021-08-25 07:56:23,402 INFO Finished backward pass...
2021-08-25 07:56:23,402 INFO Finished DPExecutor run.
2021-08-25 07:56:23,402 INFO Writing optimal schedules to auto_tvm_log/v9h-mnist-NCHW-x86_64.log successfully.
Compile...
<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7f609cd7e438>
Module(llvm, 3a73508)
Evaluate inference time cost...
Mean inference time (std dev): 0.02 ms (0.00 ms)
```

