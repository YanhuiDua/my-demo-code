import paddle
import paddle.nn as nn
import time
import paddle.profiler as profiler
import numpy as np
# paddle.set_device("npu")
paddle.set_default_dtype("float16")
# myprofiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CUSTOM_DEVICE], custom_device_types=['npu'])

# weight_attr = paddle.ParamAttr(name="weight", initializer=nn.initializer.Constant(value=1.0))
# bias_attr = paddle.ParamAttr(name="bias",initializer=nn.initializer.Constant(value=1.0))


# layer = nn.Linear(2,4, weight_attr=weight_attr, bias_attr=bias_attr)

# input = paddle.ones(shape=[3,2], dtype="float16")
# input = paddle.incubate._npu_identity(x=input, format=29) 

# output = layer(input)

# for i in range(10):
#     output = layer(input)

# start_time = time.time()
# for i in range(100):
#     if i == 50:
#        myprofiler.start()
#     output = layer(input)
#     if i == 50:
#        myprofiler.stop()
# end_time=time.time()
# print((end_time - start_time)/100)

# paddle static
paddle.enable_static()

x = np.random.random([3,2]).astype(np.float16)

main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()
with paddle.static.program_guard(main_program, startup_program):
    x_data = paddle.ones(shape=[3,2], dtype="float16")
    x_data = paddle.incubate._npu_identity(x=x_data, format=29)
    weight_attr = paddle.framework.ParamAttr(
        name="linear_weight",
        initializer=paddle.nn.initializer.Constant(value=1.0))
    bias_attr = paddle.framework.ParamAttr(
        name="linear_bias",
        initializer=paddle.nn.initializer.Constant(value=1.0))
    linear = paddle.nn.Linear(
        2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
    out=linear(x_data)

exe = paddle.static.Executor(paddle.CustomPlace("npu", 0))
exe.run(startup_program)
result = exe.run(main_program, fetch_list=[out])
print(result)

