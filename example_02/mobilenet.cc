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
#include <algorithm>
#include <iostream>

double GetCurTime(void)
{
    struct timeval tm;
    gettimeofday(&tm, 0);
    return tm.tv_usec + tm.tv_sec * 1000000;
}

cv::Mat hwc_to_chw(cv::Mat src)
{
    const int src_h = src.rows;
    const int src_w = src.cols;
    const int src_c = src.channels();

    /*            | 3 col  |
     *            ----------
     *            |  |  |  | 
     *            |  |  |  |
     *            .  .  .  .
     * (h x w)row .  .  .  .
     *            .  .  .  .
     *            |  |  |  |
     *            |  |  |  |
     *
     */
    cv::Mat hw_c = src.reshape(1, src_h * src_w);
    LOG(INFO) << "[mobilenet tvm]:hw_c:" << hw_c.channels() << "c :" << hw_c.size();

    /*       | (h x w) col |
     *       ---------------
     *       |             |
     *       ---------------
     * 3 row |             |
     *       --------------- 
     *       |             |
     */
    cv::Mat c_hw = cv::Mat();
    cv::transpose(hw_c, c_hw);
    LOG(INFO) << "[mobilenet tvm]:c_hw:" << c_hw.channels() << "c : " << c_hw.size();

    return c_hw.reshape(3, src_h);

    //dst = c_hw;
    //const std::array<int,3> dims = {src_c, src_h, src_w};
    //dst.create(3,  &dims[0], CV_MAKETYPE(src.depth(), 1));

    //LOG(INFO) << "[yolov3 tvm]:dst:" << dst.channels() << " : " << dst.rows << " : " << dst.cols;
    //LOG(INFO) << "[yolov3 tvm]:dst:" << dst.getMat().channels() << " : " << dst.getMat().rows << " : " << dst.getMat().cols;

    //cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

    //LOG(INFO) << "[yolov3 tvm]:dst_1d:" << dst_1d.channels() << " : " << dst_1d.rows << " : " << dst_1d.cols;
    //cv::transpose(hw_c, dst_1d);
    //LOG(INFO) << "[yolov3 tvm]:dst_1d:" << dst_1d.channels() << " : " << dst_1d.rows << " : " << dst_1d.cols;
    //LOG(INFO) << "[yolov3 tvm]:dst:" << dst.getMat().channels() << " : " << dst.getMat().rows << " : " << dst.getMat().cols;


}
cv::Mat LetterBox(cv::Mat img, int in_w, int in_h)
{
    int img_w = img.cols;
    int img_h = img.rows;

    int new_w = 0;
    int new_h = 0;

    if ((in_w * 1.0 / img_w) < (in_h*1.0 / img_h))
    {
       new_w = in_w;
       new_h = img_h * in_w / img_w;
    }
    else
    {
        new_h = in_h;
        new_w = img_w * in_h / img_h;
    }

    cv::Mat resize_img = cv::Mat(new_h, new_w, CV_8UC3);
    cv::resize(img, resize_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    cv::Mat boxed(in_h, in_w, CV_32FC3);
    //cv::Mat boxed(in_h, in_w, CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));

    int offset_w = (in_w - new_w) / 2;
    int offset_h = (in_h - new_h) / 2;

    for (int j = 0; j < new_h; j++)
    {
        for (int i = 0; i < new_w; i++)
        {
            boxed.at<cv::Vec3f>(j + offset_h,i + offset_w)[2] = (resize_img.at<cv::Vec3b>(j,i)[0] - 123.0) / 58.395;
            boxed.at<cv::Vec3f>(j + offset_h,i + offset_w)[1] = (resize_img.at<cv::Vec3b>(j,i)[1] - 117.0) / 57.12;
            boxed.at<cv::Vec3f>(j + offset_h,i + offset_w)[0] = (resize_img.at<cv::Vec3b>(j,i)[2] - 104.0) / 57.375;
        }
    }
    return boxed;
}

int main(int argc, char *argv[])
{
    std::string image_path, lib_path, param_path;
    int32_t loop_cnt;
    switch(argc)
    {
        case 1:
            loop_cnt   = 5;
            image_path = "./cat.png";
            lib_path   = "./mobilenet.so";
            param_path = "./mobilenet.params";
            break;

        case 2:
            loop_cnt   = atoi(argv[1]);
            image_path = "./cat.png";
            lib_path   = "./mobilenet.so";
            param_path = "./mobilenet.params";
            break;

        case 3:
            loop_cnt   = atoi(argv[1]);
            image_path = argv[2];
            lib_path   = "./mobilenet.so";
            param_path = "./mobilenet.params";
            break;

        case 4:
            loop_cnt   = atoi(argv[1]);
            image_path = argv[2];
            lib_path   = argv[3];
            param_path = "./mobilenet.params";
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

    LOG(INFO) << "[mobilenet tvm]:Image Path: " << image_path;
    LOG(INFO) << "[mobilenet tvm]:Dynamic Lib Path: " << lib_path;
    LOG(INFO) << "[mobilenet tvm]:Parameter Path: " << param_path;
    LOG(INFO) << "[mobilenet tvm]:Soft Version: V" << MNIST_VERSION;

    // read the image
    cv::Mat raw_image, resize_image;
    raw_image = cv::imread(image_path);
    LOG(INFO) << "[mobilenet tvm]:Image Type:" << raw_image.type();
    LOG(INFO) << "[mobilenet tvm]:Image Depth:" << raw_image.depth();
    if(raw_image.data == nullptr){
        LOG(INFO) << "[mobilenet tvm]:Image don't exist!";
        return 0;
    }
    else{
        resize_image = LetterBox(raw_image, 224, 224);
        resize_image = hwc_to_chw(resize_image);
        LOG(INFO) << "[mobilenet tvm]:Resize Image Type:" << resize_image.type();
        LOG(INFO) << "[mobilenet tvm]:Resize Image Depth:" << resize_image.depth();
        //cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        //gray_image.convertTo(gray_image, CV_32FC3);

        LOG(INFO) << "[mobilenet tvm]:---Load Image--";
        LOG(INFO) << "[mobilenet tvm]:Image channel: " << resize_image.channels();
        LOG(INFO) << "[mobilenet tvm]:Image size: " << resize_image.size();
        // cv::imshow("mobilenet image", gray_image);
        // cv::waitKey(0);
    }

    std::vector<float> y_output(1000);
    // create tensor
    DLTensor *x;
    DLTensor *y;
    int input_ndim  = 4;
    int output_ndim = 2;
    int64_t input_shape[4]  = {1, resize_image.channels(), resize_image.rows, resize_image.cols};
    int64_t output_shape[2] = {1, 1000};

    int dtype_code  = kDLFloat;
    int dtype_bits  = 32;
    int dtype_lanes = 1;
    int device_id   = 0;
#ifdef CPU 
    int device_type = kDLCPU;
    LOG(INFO) << "[mobilenet tvm]:--- Device Type Configure: CPU ---";
#elif OpenCL 
    int device_type = kDLOpenCL;
    LOG(INFO) << "[mobilenet tvm]:--- Device Type Configure: OPENCL ---";
#endif
    TVMByteArray params_arr;
    DLDevice dev{static_cast<DLDeviceType>(device_type), device_id};

    // allocate the array space
    TVMArrayAlloc(input_shape, input_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    TVMArrayAlloc(output_shape, output_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // load the mobilenet dynamic lib
    LOG(INFO) << "[mobilenet tvm]:---Load Dynamic Lib--";
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib_path);
    // get the mobilenet module
    tvm::runtime::Module mod = mod_dylib.GetFunction("default")(dev);

    // load the mobilenet module parameters
    LOG(INFO) << "[mobilenet tvm]:---Load Parameters--";
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
        TVMArrayCopyFromBytes(x, resize_image.data, resize_image.channels() * resize_image.rows * resize_image.cols * sizeof(float));
        set_input("input", x);
        double t2 = GetCurTime();
        run();
        TVMSynchronize(device_type, device_id, nullptr);
        double t3 = GetCurTime();
        get_output(0, y);
        TVMArrayCopyToBytes(y, y_output.data(), 1000 * sizeof(float));
        double t4 = GetCurTime();

        LOG(INFO) << "[mobilenet tvm]:---Executor[" << t << "] Time(set_input):" << t2 - t1 << "[us]";
        LOG(INFO) << "[mobilenet tvm]:---Executor[" << t << "] Time(run):" << t3 - t2 << "[us]";
        LOG(INFO) << "[mobilenet tvm]:---Executor[" << t << "] Time(get_output):" << t4 - t3 << "[us]";
    }

    std::vector<float>::iterator biggest = std::max_element(y_output.begin(), y_output.end());
    LOG(INFO) << "[mobilenet top1]:" 
              << "index = " << biggest - y_output.begin() 
              << "; score = " << *biggest;

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
