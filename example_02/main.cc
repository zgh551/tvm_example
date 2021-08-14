// tvm 
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
// opencv 
#include <opencv4/opencv2/opencv.hpp>
//#include <opencv2/opencv.hpp>
// system
#include <cstdio>
#include <fstream>
#include <sys/time.h>

double GetCurTime(void)
{
    struct timeval tm;
    gettimeofday(&tm, 0);
    return tm.tv_usec + tm.tv_sec * 1000000;
}

int main(int argc, char *argv[])
{
    std::string image_path, lib_path, param_path;
    int32_t loop_cnt;
    switch(argc)
    {
        case 1:
            loop_cnt   = 5;
            image_path = "./5.png";
            lib_path   = "./mnist.so";
            param_path = "./mnist.params";
            break;

        case 2:
            loop_cnt   = atoi(argv[1]);
            image_path = "./5.png";
            lib_path   = "./mnist.so";
            param_path = "./mnist.params";
            break;

        case 3:
            loop_cnt   = atoi(argv[1]);
            image_path = argv[2];
            lib_path   = "./mnist.so";
            param_path = "./mnist.params";
            break;

        case 4:
            loop_cnt   = atoi(argv[1]);
            image_path = argv[2];
            lib_path   = argv[3];
            param_path = "./mnist.params";
            break;

        case 5:
            loop_cnt   = atoi(argv[1]);
            image_path = argv[2];
            lib_path   = argv[3];
            param_path = argv[4];
            break;

        default:
            LOG(INFO) << "executer [loop_cnt] [data] [lib] [params]";
            return -1;
            break;
    }

    LOG(INFO) << "[mnist tvm]:Image Path: " << image_path;
    LOG(INFO) << "[mnist tvm]:Dynamic Lib Path: " << lib_path;
    LOG(INFO) << "[mnist tvm]:Parameter Path: " << param_path;
    LOG(INFO) << "[mnist tvm]:Soft Version: V" << MNIST_VERSION;

    // read the image
    cv::Mat image, gray_image;
    image = cv::imread(image_path);
    if(image.data == nullptr){
        LOG(INFO) << "[mnist tvm]:Image don't exist!";
        return 0;
    }
    else{
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        gray_image.convertTo(gray_image, CV_32FC3);

        LOG(INFO) << "[mnist tvm]:---Load Image--";
        LOG(INFO) << "[mnist tvm]:Image size: " << gray_image.rows << " X " << gray_image.cols;
        // cv::imshow("mnist image", gray_image);
        // cv::waitKey(0);
    }

    std::vector<float> y_output(10);
    // create tensor
    DLTensor *x;
    DLTensor *y;
    int input_ndim  = 4;
    int output_ndim = 2;
    int64_t input_shape[4]  = {1, 1, gray_image.rows, gray_image.cols};
    int64_t output_shape[2] = {1, 10};

    int dtype_code  = kDLFloat;
    int dtype_bits  = 32;
    int dtype_lanes = 1;
    int device_id   = 0;
#ifdef CPU 
    int device_type = kDLCPU;
    LOG(INFO) << "[mnist tvm]:--- Device Type Configure: CPU ---";
#elif OpenCL 
    int device_type = kDLOpenCL;
    LOG(INFO) << "[mnist tvm]:--- Device Type Configure: OPENCL ---";
#endif
    TVMByteArray params_arr;
    DLDevice dev{static_cast<DLDeviceType>(device_type), device_id};

    // allocate the array space
    TVMArrayAlloc(input_shape, input_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    TVMArrayAlloc(output_shape, output_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // load the mnist dynamic lib
    LOG(INFO) << "[mnist tvm]:---Load Dynamic Lib--";
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib_path);
    // get the mnist module
    tvm::runtime::Module mod = mod_dylib.GetFunction("mnist")(dev);

    // load the mnist module parameters
    LOG(INFO) << "[mnist tvm]:---Load Parameters--";
    std::ifstream params_in(param_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    // get load parameters function
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get set input data function
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    // get run function
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    // get output data function
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    
    for (int t = 0; t < loop_cnt; t++)
    {
        double t1 = GetCurTime();
        TVMArrayCopyFromBytes(x, gray_image.data, gray_image.rows * gray_image.cols * sizeof(float));
        set_input("Input3", x);
        double t2 = GetCurTime();
        run();
        TVMSynchronize(device_type, device_id, nullptr);
        double t3 = GetCurTime();
        get_output(0, y);
        TVMArrayCopyToBytes(y, y_output.data(), 10 * sizeof(float));
        double t4 = GetCurTime();

        LOG(INFO) << "[mnist tvm]:---Executor[" << t << "] Time(set_input):" << t2 - t1 << "[us]";
        LOG(INFO) << "[mnist tvm]:---Executor[" << t << "] Time(run):" << t3 - t2 << "[us]";
        LOG(INFO) << "[mnist tvm]:---Executor[" << t << "] Time(get_output):" << t4 - t3 << "[us]";
    }

    for (int i = 0; i < 10; i++)
    {
        LOG(INFO) << "[" << i << "]: " << y_output[i];
    }

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
