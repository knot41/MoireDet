The training codes are in the scripts folder, train.py

## Single-image inference on a modern PyTorch environment

The original sample contains author-specific paths and assumes physical GPU 1.
Use `script/infer.py` for a path-explicit, single-GPU-compatible inference run:

```bash
cd MoireDet
python script/infer.py \
  --input script/00002423.png \
  --checkpoint script/PSENet_100_loss0.000000.pth \
  --output outputs/official_sample_rgb.png \
  --comparison-output outputs/official_sample_comparison.png \
  --device cuda:0 \
  --color-order rgb \
  --warmup 3 \
  --runs 20
```

`rgb` matches the training data loader. Use `--color-order bgr` only to compare
against the released `sample_code.py`, which passes OpenCV BGR channels directly
to ImageNet normalization.

`--output` is the raw single-channel prediction. `--comparison-output` is a
presentation-only input/prediction visualization; it is not a comparison with a
ground-truth label, because the repository does not include one for the bundled
sample image.

For the tested modern dependency set, install PyTorch with the CUDA wheel that
matches the server image, then install `requirements-autodl.txt`.
