# 在官方 RWKV-Gradio-3 中使用 Native HF 后端

本教程把 `BlinkDL/RWKV-Gradio-3` 的模型层替换为本仓库的
`NativeRWKV7ForCausalLM`，保留原网页、tokenizer、采样和批量生成逻辑。它适合本机
验证和演示，不是 vLLM/SGLang 或生产服务框架。

## 1. 前置条件和支持环境

- Linux、NVIDIA CUDA GPU，以及已经转换完成的 RWKV-7 HF 模型目录；
- 本仓库已使用 `python -m pip install -e ".[cuda]"` 安装；
- Git、约 2 GB 环境空间，以及模型本身需要的显存和磁盘；
- 当前精确证据是 RTX 5090、FP16、官方 g1h 7.2B、Space commit `cc57df4`。

首次接入请用 0.4B/1.5B 模型确认页面，不要直接以大模型排查环境。模型转换见
[`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)。

## 2. 最小安全模型和输入

准备一个可信的本地 HF 模型目录。首次启动可以使用短输入确认进程可用；正式网页验收
必须使用下面的固定 HTML generation 提示词，并预留至少 512 个输出 token：

```text
请只输出一个完整可运行的 HTML5 文档，不要使用 Markdown 代码围栏。页面标题必须是
“RWKV-7 Native HF”，正文必须包含 id="status" 且初始文字为“ready”的状态区域，
以及 id="toggle" 的按钮。点击按钮后，状态文字必须变为“running”。CSS 和
JavaScript 必须全部内联，不得引用任何外部资源。
```

```bash
export ADAPTER=/absolute/path/to/hf-adapter
export MODEL=/absolute/path/to/rwkv7-g1d-0.4b-hf
export SPACE=/absolute/path/to/RWKV-Gradio-3-native-hf
```

`MODEL` 必须包含 `config.json`、tokenizer 文件和完整权重，不能填单个 `.pth`。
补丁会把这个目录同时交给原 Space 的启动逻辑，因此 Native 模式不需要再下载或提供
官方 `.pth`。

## 3. 可直接复制的安装和启动命令

使用独立 Space clone，不修改原目录：

```bash
git clone https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-3 "$SPACE"
git -C "$SPACE" checkout cc57df4
git -C "$SPACE" apply "$ADAPTER/examples/gradio/rwkv-gradio-3-native-hf.patch"
cp "$ADAPTER/examples/gradio/native_hf_v3a_compat.py" "$SPACE/"

python -m pip install -e "$ADAPTER[cuda]"
python -m pip install -r "$SPACE/requirements.txt"

cd "$SPACE"
APP3_BACKEND=native_hf \
APP3_HF_MODEL_PATH="$MODEL" \
APP3_MODEL_TITLE="RWKV-7 Native HF" \
APP3_DTYPE=float16 \
RWKV7_NATIVE_MODEL_BACKEND=native_graph \
python app.py
```

浏览器打开终端打印的 Gradio 地址。普通用户先使用上述保守配置。RTX 5090、
官方 g1h 7.2B、FP16 权重和 FP32 state 的精确性能配置见
[`5090_native_decode_fused_20260718`](../bench/5090_native_decode_fused_20260718/README.md)；
它不是其他型号或显卡的通用默认值。

页面默认只监听本机且不创建公开分享链接。远程机器需要临时浏览器验收时，可以显式
添加 `APP3_SHARE=1`；链接持有者可以访问页面，因此不要输入私密数据，验收后立即
停止进程。`APP3_SERVER_PORT` 可以选择其他监听端口，默认是 `7860`。

## 4. 精确且可观察的通过标准

正式 HTML generation 验收必须满足以下全部条件：

1. 进程没有 import、模型形状、CUDA graph 或 OOM 错误；
2. 对固定提示词的输出是完整 HTML5 文档，至少包含 `<!doctype html>`、`<html>`、
   `<head>`、`<body>` 和闭合的 `</html>`，且没有 Markdown 代码围栏；
3. 把生成结果作为 `.html` 在浏览器打开后，标题、`ready` 状态和按钮均可见，页面
   控制台没有 JavaScript 错误，也没有外部网络请求；
4. 点击按钮后，`#status` 的文字从 `ready` 变为 `running`；
5. 官方 v3a 与 Native HF 必须使用相同 checkpoint、提示词、采样参数和输出上限，
   两边都分别保存点击前、点击后的真实浏览器截图；
6. 页面性能标签包含 `Output ... token/s @ bsz 1`。需要验证 B8 时，八份 HTML
   输出都必须完整、可独立渲染，切回 B1 后仍能重复生成。

只出现普通文本、token/s 标签或 B1/B8 输出，只能证明网页启动与生成接口可用，不能
作为 HTML generation 验收通过。

需要记录速度时，预热后每个 batch 至少重复一次，并同时保存页面截图、标签和
`nvidia-smi` 进程显存。已有官方/Native 普通文本页面 smoke、浏览器截图和原始日志见
[`5090_gradio_native_hf_frontend_ab_20260719`](../bench/5090_gradio_native_hf_frontend_ab_20260719/README.md)。

## 5. 失败恢复和当前限制

- `No module named rwkv7_hf`：重新在当前虚拟环境执行
  `python -m pip install -e "$ADAPTER[cuda]"`。
- `requested CUDA extensions are inactive`：确认当前虚拟环境能执行
  `ninja --version`，然后重新运行 `python -m pip install -e "$ADAPTER[cuda]"`；
  严格 benchmark 不允许静默回退。
- 模型目录错误：确认 `APP3_HF_MODEL_PATH/config.json` 存在，并先运行普通
  `examples/generate.py`。
- Native 启动时仍尝试下载 `.pth`：当前 Space clone 没有应用最新补丁；重新创建
  独立 clone 后再应用补丁。
- CUDA OOM：停止进程，改用较小模型、B1 和短输出；不要同时保留多个 Gradio
  后端进程。
- 想恢复官方 Space：在独立 clone 中运行
  `git apply -R "$ADAPTER/examples/gradio/rwkv-gradio-3-native-hf.patch"`，删除
  `native_hf_v3a_compat.py`，再按官方方式启动。

早期 Gradio 路径的 `95.2/651.7 tok/s` 是保留的历史 UI 证据。当前精确 5090
配置在三次 512-token 复测中达到 B1/B8 中位 `145.06/845.57 tok/s`，相对官方
同精度 `fp32io16` state 的 `144.47/841.77 tok/s` 为 `1.0041x/1.0045x`；
扩展 active、完整 trace hash、logits cosine 和 top-1 门槛均通过。官方更低精度
fp16-state 仍为 `146.28/890.21 tok/s`，所以这些 sparse/WAG/RKV 配置继续保持
默认关闭。这些直接 decode 数据不属于 HTML generation 网页验收。

2026-07-19 的真实浏览器复测使用同一个 Space commit、g1h 7.2B、prompt 和页面采样
参数。100-token 稳定页中 Native/官方为 B1 `138.5/137.7 tok/s`、B8
`831.8/837.7 tok/s`；可读短回答页的 54 个 token 逐字一致。两边都没有继续输出提示
要求的伪代码，因此这次页面 A/B 证明后端没有引入文本差异，但不能把该回答描述为
完整遵循指令。Native B8 也仍比本次官方页面低约 `0.7%`，不能声称网页所有形状领先。

## 6. 让 AI 执行

AI 操作只从唯一入口 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 选择
`TASK_ID=gradio-native-hf`。专题教程不另存提示词，AI 必须按统一模板报告命令、
退出码、完整 HTML、点击前后渲染截图、控制台和网络请求结果、速度、显存及未通过
门槛。AI 不得用普通文本生成或直接 decode benchmark 代替 HTML generation 验收。
