# Example Two
This example using **TVM** to build the `mnist` model and deploy this model on `x86_64` , `arm` or `gpu` device.

## 1. Host Build

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
If the enter number is `1` ,the script will compile and generate the executer which run on `x86_64` device.

3. the content of executers 

In the `models` folder will  generate executers for different platforms , the content as follow:

```
models/
├── aarch64_run.tar.gz
├── data
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── 5.png
│   ├── 6.png
│   ├── 7.png
│   ├── 8.png
│   └── 9.png
├── opencl_run.tar.gz
└── x86_64
    ├── 5.png
    ├── mnist.params
    ├── mnist.so
    └── mnist_test

```
The `data` folder contain some test data. The `aarch64_run.tar.gz` and `opencl_run.tar.gz` file can copy to the target device to running. In the PC host, you can enter into `x86_64` folder to run `./mnist_test`.

>> note: make sure the configure of python environment which contain tvm and AI fronted frame is ok.

## 2. Device Run

### 2.1. Dependent library
- **OpenCV**
1. libopencv_core.so
2. libopencv_imgcodecs.so
3. libopencv_imgproc.so

in the `depend` folder include the `opencv_deploy.run` file, if target device don't contain this dynamic library, you should copy this file to target device and execute command as follow:

```bash
$ chmod 775 opencv_deploy.run
$ ./opencv_deploy.run
```

this will auto deploy the dynamic library to the `/usr/sdrv/opencv` folder.

- **TVM**
1. libtvm-runtime.so

In the `tvm` folder, run `./build.sh deploy` will generate the `tvm_runtime_deploy.run` file in `build_aarch64`folder,  if target device don't contain this dynamic library, you should copy this file to target device and execute command as follow:

```shell
$ chmod 775 tvm_runtime_deploy.run
$ ./tvm_runtime_deploy.run
```

This will auto deploy the `libtvm-runtime.so` to `/usr/sdrv/tvm` folder.

### 2.2. Copy Runnable Package

- copy

In the `models` folder, copy `aarch64_run.tar.gz` or `opencl_run.tar.gz`  file to the target device.

- unzip

```shell
$ tar -zxvf aarch64_run.tar.gz
$ tar -zxvf opencl_run.tar.gz
```

- run

```shell
$ cd aarch64_run/aarhc64
$ ./mnist_test
```

or

```shell
$ cd opencl_run/opencl
$ ./mnist_test
```


### 2.3. Running

The executable file with four optional parameters, we can running with the numbers of parameters from 0 to 4, the optional parameters as follow:

```shell
$ executer [loop_cnt] [data] [lib] [params] 
```

For example:

1. zero parameters mode

```shell
$ ./minst_test
```

This mode will auto load the `data`, `lib` and `params` under the current folder, the default loop count is **5**.

2. one parameters mode

```shell
$ ./mnist_test 100
```

This mode can set the loop count as need.

3. two parameters mode

```shell
$ ./mnist_test 100 ../data/4.png 
```
This mode can set the loop count and input data as need.

4. tree parameters mode

```
$ ./mnist_test 100 ../data/6.png ./mnist.so
```
This mode can set the loop count ,input data  and model dynamic library as need.


5. four parameters mode

```shell
$ ./mnist_test 100 ../data/8.png ./mnist.so ./mnist.params 
```

If select four parameters mode, you can specify the input data and the loop count. The dynamic library and parameter of model can also configure.

### 2.4 Result

```shell
[21:57:32] main.cc:42: [mnist tvm]:Image Path: ./5.png                                       
[21:57:32] main.cc:43: [mnist tvm]:Dynamic Lib Path: ./mnist.so
[21:57:32] main.cc:44: [mnist tvm]:Parameter Path: ./mnist.params
[21:57:32] main.cc:45: [mnist tvm]:Soft Version: V1.1.2
[21:57:32] main.cc:58: [mnist tvm]:---Load Image--
[21:57:32] main.cc:59: [mnist tvm]:Image size: 28 X 28
[21:57:32] main.cc:79: [mnist tvm]:--- Device Type Configure: CPU ---
[21:57:32] main.cc:92: [mnist tvm]:---Load Dynamic Lib--
[21:57:32] main.cc:98: [mnist tvm]:---Load Parameters--
[21:57:32] main.cc:130: [mnist tvm]:---Executor[0] Time(set_input):2[us]
[21:57:32] main.cc:131: [mnist tvm]:---Executor[0] Time(run):154[us]
[21:57:32] main.cc:132: [mnist tvm]:---Executor[0] Time(get_output):2[us]
[21:57:32] main.cc:130: [mnist tvm]:---Executor[1] Time(set_input):1[us]
[21:57:32] main.cc:131: [mnist tvm]:---Executor[1] Time(run):132[us]
[21:57:32] main.cc:132: [mnist tvm]:---Executor[1] Time(get_output):1[us]
[21:57:32] main.cc:130: [mnist tvm]:---Executor[2] Time(set_input):2[us]
[21:57:32] main.cc:131: [mnist tvm]:---Executor[2] Time(run):92[us]
[21:57:32] main.cc:132: [mnist tvm]:---Executor[2] Time(get_output):0[us]
[21:57:32] main.cc:130: [mnist tvm]:---Executor[3] Time(set_input):1[us]
[21:57:32] main.cc:131: [mnist tvm]:---Executor[3] Time(run):92[us]
[21:57:32] main.cc:132: [mnist tvm]:---Executor[3] Time(get_output):0[us]
[21:57:32] main.cc:130: [mnist tvm]:---Executor[4] Time(set_input):1[us]
[21:57:32] main.cc:131: [mnist tvm]:---Executor[4] Time(run):98[us]
[21:57:32] main.cc:132: [mnist tvm]:---Executor[4] Time(get_output):0[us]
[21:57:32] main.cc:137: [0]: -2702.9
[21:57:32] main.cc:137: [1]: -2044.03
[21:57:32] main.cc:137: [2]: -1144.48
[21:57:32] main.cc:137: [3]: 3431.76
[21:57:32] main.cc:137: [4]: -2802.41
[21:57:32] main.cc:137: [5]: 4449.7
[21:57:32] main.cc:137: [6]: -2858.96
[21:57:32] main.cc:137: [7]: 127.987
[21:57:32] main.cc:137: [8]: 912.245
[21:57:32] main.cc:137: [9]: 528.989
```

