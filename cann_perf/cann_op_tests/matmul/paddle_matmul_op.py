import paddle
import paddle.nn as nn
import time
import paddle.profiler as profiler

paddle.set_device("npu")
myprofiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CUSTOM_DEVICE], custom_device_types=['npu'])
x = paddle.rand(shape=[5],dtype="float32")

# print(x.shape)
y = paddle.rand(shape=[2,3,5,4],dtype="float32")
 
# print(y.shape)

z = paddle.matmul(x, y)
print(z.shape)

# for i in range(20):
#     paddle.matmul(x, y)

# start_time = time.time()
# for i in range(50):
#     if i == 25:
#         myprofiler.start()
#     paddle.matmul(x, y)
#     if i == 25:
#         myprofiler.stop()
# end_time=time.time()
# print((end_time - start_time)/50.0)

# print("run_time: ", end_time -  start_time)

