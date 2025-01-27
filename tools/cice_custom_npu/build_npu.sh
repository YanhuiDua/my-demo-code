#!/bin/bash
set -xe

##### global environment, delete from scirpt and add into ci config #####

export WORKSPACE=/workspace/npu-dev
export CACHE_ROOT=/workspace/npu-dev

export PADDLE_BRANCH=develop
export PADDLE_COMMIT=develop
export PADDLE_VERSION=0.0.0

export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann600-$(uname -m)-gcc82

export whl_package=paddle-device/develop/npu
export tgz_package=paddle-device/develop/npu

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
export ftp_proxy=${proxy}
export no_proxy=bcebos.com
set -x

mkdir -p ${WORKSPACE}
mkdir -p ${CACHE_ROOT}
rm -rf ${WORKSPACE}/output/*

cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*
rm -rf output*

# git clone PaddleCustomDevice
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone --depth=200 --recursive https://github.com/PaddlePaddle/PaddleCustomDevice.git
cd PaddleCustomDevice
# sync submodule
git submodule sync
git submodule update --init --recursive
# show git log history
git log --pretty=oneline -20

# prepare cache dir
source_dir="${WORKSPACE}/PaddleCustomDevice"
cache_dir="${CACHE_ROOT}/.cache"
ccache_dir="${CACHE_ROOT}/.ccache"

# start ci test in container
set -ex
docker pull ${PADDLE_DEV_NAME}
docker run --rm -i \
  --privileged --pids-limit 409600 --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${source_dir}:/paddle -w /paddle \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "no_proxy=bcebos.com" \
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.version.commit)"

bash -x backends/npu/tools/pr_ci_npu.sh;EXCODE=$?

if [[ $EXCODE -eq 0 ]];then
    echo "Congratulations!  Your PR passed the CI."
elif [[ $EXCODE -eq 4 ]];then
    echo "Sorry, your code style check failed."
elif [[ $EXCODE -eq 6 ]];then
    echo "Sorry, your pr need to be approved."
elif [[ $EXCODE -eq 7 ]];then
    echo "Sorry, build failed."
elif [[ $EXCODE -eq 8 ]];then
    echo "Sorry, some tests failed."
elif [[ $EXCODE -eq 9 ]];then
    echo "Sorry, coverage check failed."
fi

exit $EXCODE
'

mkdir -p ${WORKSPACE}/output
cp ${source_dir}/backends/npu/build/dist/paddle_custom_npu*.whl ${WORKSPACE}/output

wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz --no-check-certificate
tar xf bos_new.tar.gz -C ${WORKSPACE}/output

# Install dependency
python3 -m pip install bce-python-sdk -i http://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com

# Upload paddlepaddle-rocm whl package to paddle-device/develop/dcu1
cd ${WORKSPACE}/output
for file_whl in `ls *.whl` ;do
  python3 BosClient.py ${file_whl} ${whl_package}
done
