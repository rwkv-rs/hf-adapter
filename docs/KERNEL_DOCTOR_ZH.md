# 运行环境与算子策略检查

英文版：[`KERNEL_DOCTOR.md`](KERNEL_DOCTOR.md)

安装补丁版后、加载大模型前，先运行只读检查：

```bash
python -m pip install "rwkv7-hf==0.8.0"
```

```bash
python -m rwkv7_hf.doctor
```

安装 wheel 后也可以使用等价命令：

```bash
rwkv7-hf-doctor
```

报告会显示 Python、PyTorch、Transformers、CUDA/ROCm、Triton、NVCC、Ninja、
可见加速设备、可选 `rwkv7-kernels` manifest 兼容性、硬件分类、默认算子策略和
编译缓存目录。检查成功时最后显示 `RESULT: READY`。

报告中的算子是**候选路线**，不代表该算子已经实际执行。最终路线还取决于模型
形状、dtype、batch、序列长度、可选依赖和环境变量。本命令不会下载权重、编译扩展、
捕获 CUDA Graph、运行性能测试或安装算子 wheel；存在精确匹配时再执行
`rwkv7-hf-kernels recommend` 和 `rwkv7-hf-kernels install`。

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
  扩展无法构建，但 Torch/Triton 路线可能仍然可用。如果兼容的预编译 wheel 已
  就绪，则 wheel 内包含的扩展不再需要本地编译器。
- **预编译算子缺失或不兼容：** 运行 `rwkv7-hf-kernels status` 和
  `rwkv7-hf-kernels recommend`。只有 Python、Torch、CUDA、ABI 和 compute
  capability 全部匹配公开哈希索引时才会提供安装。
  详见 [`KERNEL_WHEELS_ZH.md`](KERNEL_WHEELS_ZH.md)。
- **PyTorch CUDA 二进制不支持当前计算能力：** 需要安装覆盖该 GPU 架构的
  PyTorch；这是硬性 `RESULT: FAIL`，否则后续 CUDA tensor 执行也会失败。
- **没有精确验证的硬件档案：** 当前设备使用保守策略，直到真实卡验收完成。

完整依赖和模型目录检查继续使用
[`examples/check_environment.py`](../examples/check_environment.py)。本命令专门检查
加速设备和算子策略。
