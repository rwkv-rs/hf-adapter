# Tutorial image sources

The six PNG files in this directory are deterministic 1200x675 documentation
diagrams rendered from [`source.html`](source.html). Keep commands in the images
short; the adjacent Markdown guide is the copyable source of truth.

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
