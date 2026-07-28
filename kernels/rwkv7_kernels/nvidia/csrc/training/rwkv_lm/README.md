# RWKV-LM train_temp CUDA sources

These sources are vendored from `BlinkDL/RWKV-LM` commit
`e6f74b63a06e08606d130043599d218209628bad`, under
`RWKV-v7/train_temp/cuda/`.

The upstream files are licensed under Apache-2.0; the complete upstream
license is preserved in this directory as `LICENSE`. Local integration and
runtime validation live in `rwkv7_hf/train_temp_cuda.py`.

The backend is opt-in and currently targets Linux CUDA, BF16, RWKV-7 head
size 64, and dense sequence lengths divisible by 16. These files are not
compiled during normal HF inference or training.
