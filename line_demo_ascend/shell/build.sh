#!/bin/bash
cur_dir=$(pwd)

build_dir=$cur_dir/build
mkdir -p $build_dir
cd $build_dir

 cmake -DCMAKE_VERBOSE_MAKEFILE=ON \
             -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
             -DCMAKE_BUILD_TYPE=Release \
             ..
make

cd -
echo "ls -l $build_dir"
ls -l $build_dir