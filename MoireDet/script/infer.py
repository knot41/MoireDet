#!/usr/bin/env python3
"""Run MoireDet on one image with a released-format checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from lib.models import get_model  # noqa: E402


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Predict a single-channel moire edge map for one image.'
    )
    parser.add_argument('--input', required=True, type=Path, help='Input image path.')
    parser.add_argument(
        '--checkpoint', required=True, type=Path, help='MoireDet checkpoint path.'
    )
    parser.add_argument('--output', required=True, type=Path, help='Output PNG path.')
    parser.add_argument(
        '--comparison-output',
        type=Path,
        help='Optional side-by-side PNG containing the input and predicted edge map.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device such as auto, cpu, cuda, or cuda:0 (default: auto).',
    )
    parser.add_argument(
        '--color-order',
        choices=('rgb', 'bgr'),
        default='rgb',
        help=(
            'Channel order before ImageNet normalization. rgb matches the training '
            'loader; bgr reproduces the released sample_code.py behavior.'
        ),
    )
    parser.add_argument('--size', type=int, default=320, help='Square input size.')
    parser.add_argument(
        '--warmup', type=int, default=1, help='Unmeasured warmup forwards (default: 1).'
    )
    parser.add_argument(
        '--runs', type=int, default=1, help='Measured forwards to average (default: 1).'
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is false.')
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f'CUDA device index {device_index} is unavailable; '
                f'found {torch.cuda.device_count()} device(s).'
            )
    return device


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a checkpoint dictionary containing state_dict.')
    if 'config' not in checkpoint:
        raise ValueError('Expected the released checkpoint to contain its config.')
    return checkpoint


def build_model(checkpoint: dict) -> torch.nn.Module:
    config = copy.deepcopy(checkpoint['config'])
    model_name = config.get('arch', {}).get('model_name')
    if model_name != 'TripleBranchWithSpecificConv':
        raise ValueError(
            f'Unexpected model_name {model_name!r}; expected TripleBranchWithSpecificConv.'
        )

    # The full checkpoint replaces every model parameter, so downloading a
    # separate ImageNet ResNet18 state dict during construction is unnecessary.
    config['arch']['args']['pretrained'] = False
    model = get_model(config)

    state_dict = {
        key.removeprefix('module.'): value
        for key, value in checkpoint['state_dict'].items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model


def preprocess(path: Path, size: int, color_order: str) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'OpenCV could not read the input image: {path}')
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if color_order == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor


def save_comparison(
    input_path: Path, prediction: np.ndarray, output_path: Path, size: int
) -> None:
    """Save a presentation image; this does not create a ground-truth label."""
    original = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f'OpenCV could not read the input image: {input_path}')
    original = cv2.resize(original, (size, size), interpolation=cv2.INTER_LINEAR)
    prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

    title_height = 36
    canvas = np.zeros((size + title_height, size * 2, 3), dtype=np.uint8)
    canvas[title_height:, :size] = original
    canvas[title_height:, size:] = prediction_bgr
    cv2.putText(
        canvas,
        'Input image',
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        'Predicted moire edge map',
        (size + 12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (size, 0), (size, size + title_height - 1), (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f'OpenCV failed to write: {output_path}')


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError('--warmup must be >= 0 and --runs must be >= 1.')
    if args.size < 1:
        raise ValueError('--size must be >= 1.')
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint).to(device).eval()
    image = preprocess(args.input, args.size, args.color_order).to(device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        elapsed_ms = []
        prediction = None
        for _ in range(args.runs):
            started = time.perf_counter()
            predictions, _ = model(image)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
            prediction = predictions[0][0, 0]

    assert prediction is not None

    prediction = prediction.float().cpu().numpy()
    output = np.clip(prediction, 0.0, 255.0).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f'OpenCV failed to write: {args.output}')
    if args.comparison_output is not None:
        save_comparison(args.input, output, args.comparison_output, args.size)

    metadata = {
        'input': str(args.input.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'output': str(args.output.resolve()),
        'comparison_output': (
            str(args.comparison_output.resolve())
            if args.comparison_output is not None
            else None
        ),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'model_name': checkpoint['config']['arch']['model_name'],
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'color_order': args.color_order,
        'input_shape': list(image.shape),
        'output_shape': list(output.shape),
        'output_min_raw': float(prediction.min()),
        'output_max_raw': float(prediction.max()),
        'output_mean_raw': float(prediction.mean()),
        'warmup_runs': args.warmup,
        'measured_runs': args.runs,
        'elapsed_ms_mean': float(np.mean(elapsed_ms)),
        'elapsed_ms_min': float(np.min(elapsed_ms)),
        'elapsed_ms_max': float(np.max(elapsed_ms)),
        'peak_gpu_memory_mib': (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == 'cuda'
            else None
        ),
        'torch_version': torch.__version__,
        'cuda_runtime': torch.version.cuda,
    }
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
