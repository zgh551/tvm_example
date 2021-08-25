#! /bin/bash
#
# build.sh
# Distributed under terms of the MIT license.
#
# x86_64 aarch64 opencl
echo "---| number | target type|"
echo "-->   [1]   |   x86_64   |"
echo "-->   [2]   |   aarch64  |"
echo "-->   [3]   |   opencl   |"
echo "-->   [4]   |   all      |"
echo -n "Enter target type number: "
read n

echo "---| number | models         |"
echo "-->   [1]   | mnist          |"
echo "-->   [2]   | mobilenet      |"
echo "-->   [3]   | op9_dla_onnx   |"
echo "-->   [4]   | op9_dla_tflite |"
echo -n "Enter model number: "
read m


x86_64_func(){
    case $m in
    1)
        echo "for mnist"
        python3 python/rpc_deploy.py x86_64 mnist
        ;;
    2)
        echo "for mobilenet"
        python3 python/rpc_deploy.py x86_64 mobilenet
        ;;
    3)
        echo "for op9_dla_onnx"
        python3 python/rpc_deploy.py x86_64 op9_dla_onnx
        ;;
    4)
        echo "for op9_dla_tflite"
        python3 python/rpc_deploy.py x86_64 op9_dla_tflite
        ;;
    *)
        echo "None model" 
        ;;
    esac
}
        
aarch64_func(){
    case $m in
    1)
        echo "for mnist"
        python3 python/rpc_deploy.py aarch64 mnist
        ;;
    2)
        echo " for mobilenet"
        python3 python/rpc_deploy.py aarch64 mobilenet
        ;;
    3)
        echo "for op9_dla_onnx"
        python3 python/rpc_deploy.py aarch64 op9_dla_onnx
        ;;
    4)
        echo "for op9_dla_tflite"
        python3 python/rpc_deploy.py aarch64 op9_dla_tflite
        ;;
    *)
        echo "None model" 
        ;;
    esac
}
        
opencl_func(){
    case $m in
    1)
        echo "for mnist"
        python3 python/rpc_deploy.py opencl mnist
        ;;
    2)
        echo " for mobilenet"
        python3 python/rpc_deploy.py opencl mobilenet
        ;;
    3)
        echo "for op9_dla_onnx"
        python3 python/rpc_deploy.py opencl op9_dla_onnx
        ;;
    4)
        echo "for op9_dla_tflite"
        python3 python/rpc_deploy.py opencl op9_dla_tflite
        ;;
    *)
        echo "None model" 
        ;;
    esac
}


case $n in
1)
    echo -n "x86_64 build "
    x86_64_func
    ;;
2)
    echo -n "aarch64 build "
    aarch64_func
    ;;
3)
    echo -n "opencl build "
    opencl_func
    ;;
4)
    echo -n "all build "
    x86_64_func
    aarch64_func
    opencl_func
    ;;
*)
    echo "invalid build "
    ;;
esac

