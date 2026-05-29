# 运行环境与算子策略检查

英文版：[`KERNEL_DOCTOR.md`](KERNEL_DOCTOR.md)

安装补丁版后、加载大模型前，先运行只读检查：

```bash
python -m pip install "rwkv7-hf==0.7.1"
```

```bash
python -m rwkv7_hf.doctor
```

安装 wheel 后也可以使用等价命令：

```bash
rwkv7-hf-doctor
```

报告会显示 Python、PyTorch、Transformers、CUDA/ROCm、Triton、NVCC、Ninja、
可见加速设备、硬件分类、默认算子策略和编译缓存目录。检查成功时最后显示
`RESULT: READY`。

报告中的算子是**候选路线**，不代表该算子已经实际执行。最终路线还取决于模型
形状、dtype、batch、序列长度、可选依赖和环境变量。本命令不会下载权重、编译扩展、
捕获 CUDA Graph 或运行性能测试。

## 检查指定设备

默认报告所有可见 CUDA 设备。只检查一张卡：

```bash
python -m rwkv7_hf.doctor --device cuda:1
```

也可以指定 `--device mps` 或当前 PyTorch 能识别的其他设备。

## 输出机器可读证据

```bash
python -m rwkv7_hf.doctor \
  --json \
  --output rwkv7-doctor.json
```

报告算子未启用的问题前，请附上这个 JSON。它不会主动收集模型权重或认证 token，
但会包含本机编译器与缓存路径；公开分享前先检查内容。

## 警告含义

- **Triton 不可用：** Triton 候选路线不能运行，但兼容 Torch 路线仍可工作。
- **CUDA 扩展工具链不完整：** 没有同时找到 NVCC 和 Ninja；依赖它们的 JIT
  扩展无法构建，但 Torch/Triton 路线可能仍然可用。
- **PyTorch CUDA 二进制不支持当前计算能力：** 需要安装覆盖该 GPU 架构的
  PyTorch；这是硬性 `RESULT: FAIL`，否则后续 CUDA tensor 执行也会失败。
- **没有精确验证的硬件档案：** 当前设备使用保守策略，直到真实卡验收完成。

完整依赖和模型目录检查继续使用
[`examples/check_environment.py`](../examples/check_environment.py)。本命令专门检查
加速设备和算子策略。
