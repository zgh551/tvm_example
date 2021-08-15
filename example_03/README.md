# Example Tree
This example introduce one way to use `tvm rpc tracker server` to tune and evaluate the performance of  the **mnist** model  on `x86_64` , `arm` or `gpu` device.

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

Open the `{example_base_dir}/python/tune_schedule.py` file and replace  `rpc_host="192.168.105.70"` to pc host actual `ip` address. `rpc_port=9190` set as the host tracker server port and the `device_key=v9h` set as the device name.

### 2.3. Running Script

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

### 2.3 Result

```shell
---| number | target type|                                               
-->   [1]   |   x86_64   |                                               
-->   [2]   |   aarch64  |                                               
-->   [3]   |   opencl   | 
-->   [4]   |   all      |
Enter target type number: 1
x86_64 build
target device: v9h
rpc_host: 192.168.0.109:9190
shape_dict:  {'Input3': (1, 1, 28, 28)}
device: llvm -keys=cpu -link-params=0
log file: mnist-NCHW-B1-llvm-C200-T21-08-14-21-44.json
Begin tuning...
Get devices for measurement successfully!
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |            - |              - |      0 |
|    1 |            - |              - |      0 |
|    2 |            - |              - |      0 |
|    3 |            - |              - |      0 |
|    4 |            - |              - |      0 |
-------------------------------------------------
Estimated total latency: - ms   Trials: 0       Used time : 0 s Next ID: 0
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Generate Sketches               #s: 5
Sample Initial Population       #s: 789 fail_ct: 835    Time elapsed: 3.27
GA Iter: 0      Max score: 0.9993       Min score: 0.8937       #Pop: 80        #M+: 0  #M-: 0
GA Iter: 4      Max score: 0.9999       Min score: 0.9897       #Pop: 80        #M+: 1380       #M-: 77
EvolutionarySearch              #s: 80  Time elapsed: 13.79
----------------------------------------------------------------------
```

