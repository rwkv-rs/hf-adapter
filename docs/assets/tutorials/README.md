# Tutorial image sources

The deterministic workflow diagrams in this directory are rendered at 1200x675
from [`source.html`](source.html). Files 11 and 12 are current screenshots of the
official Hugging Face model list and GitHub tokenizer page. Keep commands in the
images short; the adjacent Markdown guide is the copyable source of truth.

To update an image:

1. Edit the corresponding `<section>` in `source.html`.
2. Open `source.html?slide=<section-id>` in a 1200x675 browser viewport.
3. Capture the viewport at device scale 1 with scrollbars hidden.
4. Run `python -m pytest tests/test_user_quickstart.py -q` to verify the PNG
   signature, dimensions, links, and workflow command coverage.

Section and output mapping:

| Section id | Output |
|---|---|
| `first-run` | `01-first-run.png` |
| `speculative` | `02-speculative-decoding.png` |
| `training` | `03-single-gpu-training.png` |
| `multi-gpu-inference` | `04-multi-gpu-inference.png` |
| `multi-gpu-training` | `05-multi-gpu-training.png` |
| `ai-assistant` | `06-ai-assisted-setup.png` |
| `inference-cache` | `07-inference-and-cache.png` |
| `training-ecosystem` | `08-training-ecosystem.png` |
| `quantization` | `09-quantization-paths.png` |
| `apple-deployment` | `10-apple-deployment.png` |
| official Hugging Face page | `11-huggingface-model-download.jpg` |
| official GitHub page | `12-github-tokenizer-download.jpg` |
| `download-layout` | `13-download-directory-layout.png` |
| `backend-choice` | `14-backend-choice.png` |
| `troubleshooting` | `15-first-error-recovery.png` |
| `ai-task-router` | `16-ai-task-router.png` |
