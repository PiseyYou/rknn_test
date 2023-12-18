# 一，项目说明

该工程测试rknn的模型及相关流程，主要包括2个功能模块

1，onnx2rknn_debug.py，测试onnx转rknn后，从rknn输出的结果

2，test_onnx_demo.py，测试onnx原始的输出结果



# 二，使用说明

1，拷贝工程路径

2，按照rknn依赖库，本案例使用的rknn-toolkit2版本为: 1.4.0-22dcfef4

```shell
#pc端
python test_onnx_demo.py #测试onnx原始的输出结果
python onnx2rknn_debug.py #测试rknn输出结果
```

