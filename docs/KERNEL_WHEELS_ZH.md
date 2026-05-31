# 预编译 CUDA 算子 wheel

英文版：[`KERNEL_WHEELS.md`](KERNEL_WHEELS.md)

`rwkv7-hf` 继续作为可移植的 Transformers 包。它包含模型实现、路由策略、Triton
代码、保守 PyTorch 回退，以及旧版按需 JIT 路线所需的 CUDA 源码。可选的
`rwkv7-kernels` wheel 则为一个**精确运行环境**提供已经编译完成的 CUDA 扩展。

## 普通用户流程

先安装运行库并检查环境：

```bash
python -m pip install "rwkv7-hf==0.8.0"
rwkv7-hf-kernels status
rwkv7-hf-kernels recommend
rwkv7-hf-doctor
```

基础 PyPI 安装有意不自动安装 GPU wheel。如果当前 Linux NVIDIA 环境存在经过
验证的精确 wheel，`recommend` 会列出唯一构建，再用一条命令安装：

```bash
rwkv7-hf-kernels install
rwkv7-hf-doctor
```

安装器会同时匹配：

- Python 主次版本和 CPython ABI；
- 操作系统与机器架构；
- PyTorch 主次版本和 libstdc++ ABI；
- PyTorch CUDA runtime；
- GPU compute capability；
- `rwkv7-hf` 兼容版本范围。

安装地址带有索引中记录的 SHA256。任何一项不完全相符都不会安装“相近版本”。
没有匹配 wheel 时，原有 JIT、Triton 或 PyTorch 正确回退仍然可用。

安装后不需要给模型设置算子。默认 `RWKV7_KERNELS_MODE=auto` 会在每个新进程重新
验证 manifest，并自动优先选择兼容预编译扩展，再进入 JIT 或可移植回退；过期或
不兼容的包不会被导入。

## 首批已验证矩阵

| 构建路线 | 实测设备 | 包含的扩展 |
|---|---|---|
| CPython 3.11、Torch 2.5、CUDA 12.4、`sm_70` | Tesla V100-PCIE-32GB | FP16 recurrence、SM70 linear、SM70 W/A/G/V、SM7x quant、sparse FFN |
| CPython 3.11、Torch 2.6、CUDA 12.4、`sm_89` | NVIDIA GeForce RTX 4080 | FP16 recurrence、Ada W/A/G/V、sparse FFN |

构建器也描述了 SM75、SM80、SM86、SM90、SM120 源码路线，但“NVCC 可以编译”
不等于“已验证发布”。新增路线必须在同卡完成导入、正确性、实际路由和公开安装验收。

## 选择与回退

`RWKV7_KERNELS_MODE` 控制原生扩展边界：

| 值 | 行为 |
|---|---|
| `auto` | 优先兼容的预编译 wheel，否则保留按需 JIT；这是默认值。 |
| `prebuilt` | 强制预编译扩展，禁止 JIT 回退；发布验收使用此模式。 |
| `jit` | 忽略预编译包，只使用原来的按需 JIT。 |
| `portable` | 同时关闭二进制扩展和 JIT；继续使用 Triton、tensor 或 eager 回退。 |

所有接入的算子统一按下面顺序加载：

```text
兼容的 rwkv7-kernels 二进制
  -> 按需 JIT（只在 auto/jit 下）
  -> 原有正确的 Triton/tensor/PyTorch 回退
```

模型实际运行后可查看真实路线：

```python
report = model.rwkv7_runtime_report()
print(report["last_prefill_backend"])
print(report["last_decode_backend"])
print(report["kernels"]["extensions"])
```

Doctor 只检查 manifest，不导入二进制；模型报告记录算子真正被预编译包或 JIT
成功加载后的结果。

## 一条命令公开模型验收

```bash
rwkv7-hf-smoke \
  --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0 \
  --device cuda \
  --output rwkv7-smoke.json
```

命令会下载并加载最小公开模型，执行 Prefill 和贪心 Decode，检查有限 logits，记录
实际 Prefill/Decode/算子路线、CUDA 峰值分配，并最终打印 `RESULT: PASS`。时间字段
只用于安装 smoke，不作为正式性能结论。

## 构建和发布

必须使用 wheel 所对应的 Python、PyTorch 与 CUDA 环境：

```bash
export CUDA_HOME=/path/to/cuda-12.4
export PYTHON_BIN=/path/to/target/python
export RWKV7_KERNEL_ARCH_LIST=8.9
export RWKV7_KERNEL_SOURCE_COMMIT="$(git rev-parse HEAD)"
export OUT_DIR=/path/to/output
bash scripts/build_kernel_wheel.sh
```

构建脚本从 `rwkv7_hf` 的权威源码字符串生成临时 C++/CUDA 文件，编译精确架构
模块，写入 `_manifest.json`，并运行 `scripts/inspect_kernel_wheel.py`。临时源码和
二进制不会提交进 Git。

手动触发的 `kernel-wheels` 工作流会在带标签的 SM70/SM89 硬件 runner 上完成相同
构建；随后隔离安装 adapter 和算子 wheel，强制 `RWKV7_KERNELS_MODE=prebuilt`，
运行同卡算子测试，生成哈希索引，并可把已验证产物上传到已有 GitHub Release。
