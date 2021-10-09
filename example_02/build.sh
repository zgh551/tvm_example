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

if [ ! -d models ]; then
    mkdir models
else 
    rm -rf models/*
fi
cp -r data/ models/

x86_64_func(){
    mkdir models/x86_64
    python3 python/model_build.py x86_64

    if [ ! -d build_x86_64 ]; then
        mkdir build_x86_64
    else 
        rm -rf build_x86_64/*
    fi
    cd build_x86_64
    cmake ..
    make -j$(nproc)
    cp mobilenet_run ../models/x86_64
    cd ..
    rm -rf build_x86_64
    cp data/cat.png models/x86_64
}
        
aarch64_func(){
    mkdir models/aarch64
    python3 python/model_build.py aarch64

    if [ ! -d build_aarch64 ]; then
        mkdir build_aarch64
    else
        rm -rf build_aarch64/*
    fi
    cd build_aarch64
    cmake -DCMAKE_SYSTEM_NAME=Linux \
          -DCMAKE_SYSTEM_VERSION=1 \
          -DMACHINE_NAME=aarch64 \
          -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
          -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
          -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
          -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
          -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH ..

    make -j$(nproc)
    cp mobilenet_run ../models/aarch64
    cd ..
    rm -rf build_aarch64
    cp data/cat.png models/aarch64
    cd models
    mkdir aarch64_run
    cp -r data/ aarch64_run/
    mv aarch64/ aarch64_run/
    tar -zcpf aarch64_run.tar.gz aarch64_run/
    rm -rf aarch64_run
    cd ..
}
        
opencl_func(){
    mkdir models/opencl
    python3 python/model_build.py opencl

    if [ ! -d build_opencl ]; then
        mkdir build_opencl
    else
        rm -rf build_opencl/*
    fi
    cd build_opencl
    cmake -DCMAKE_SYSTEM_NAME=Linux \
          -DCMAKE_SYSTEM_VERSION=1 \
          -DTARGET_DEVICE_TYPE=OpenCL \
          -DMACHINE_NAME=aarch64 \
          -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
          -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
          -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
          -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
          -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH ..

    make -j$(nproc)
    cp mobilenet_run ../models/opencl
    cd ..
    rm -rf build_opencl
    cp data/cat.png models/opencl
    cd models
    mkdir opencl_run
    cp -r data/ opencl_run/
    mv opencl/ opencl_run/
    tar -zcpf opencl_run.tar.gz opencl_run/
    rm -rf opencl_run
    cd ..
}


case $n in
1)
    echo "x86_64 build"
    x86_64_func
    ;;
2)
    echo "aarch64 build"
    aarch64_func
    ;;
3)
    echo "opencl build"
    opencl_func
    ;;
4)
    echo "all build"
    x86_64_func
    aarch64_func
    opencl_func
    ;;
*)
    echo "invalid"
    ;;
esac




