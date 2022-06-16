#!/bin/bash

TARGET_EXE=${1:-BroadcastToD}

echo "----------- buiding target : ${TARGET_EXE} --------------"

build_dir="$(pwd)/build"
if [ -d ${build_dir} ];then
  rm -rf ${build_dir}
fi

# update TARGET_EXE for the operator
# TARGET_EXE=Add
# TARGET_EXE=Resize
# TARGET_EXE=ResizeD
# TARGET_EXE=ResizeNearestNeighborV2
# TARGET_EXE=BroadcastTo
# TARGET_EXE=BroadcastToD

# build
mkdir build && cd build
cmake -DTARGET_EXE=${TARGET_EXE} -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make

# run
echo "-------------- start running op : ${TARGET_EXE} --------------"
./${TARGET_EXE}
