#!/usr/bin/env python3
"""Run MoireDet frame by frame and write an original/prediction side-by-side MP4."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from infer import (  # noqa: E402
    build_model,
    load_checkpoint,
    preprocess_image,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a side-by-side original/predicted moire video.'
    )
    parser.add_argument('--input', required=True, type=Path, help='Input video path.')
    parser.add_argument(
        '--checkpoint', required=True, type=Path, help='MoireDet checkpoint path.'
    )
    parser.add_argument('--output', required=True, type=Path, help='Output MP4 path.')
    parser.add_argument(
        '--metadata-output',
        type=Path,
        help='Optional JSON metadata path; defaults to output path with a .json suffix.',
    )
    parser.add_argument(
        '--device', default='auto', help='Device such as auto, cpu, or cuda:0.'
    )
    parser.add_argument(
        '--color-order',
        choices=('rgb', 'bgr'),
        default='rgb',
        help='Input channel order before normalization; rgb matches training.',
    )
    parser.add_argument('--size', type=int, default=320, help='Square model input size.')
    parser.add_argument(
        '--display-height',
        type=int,
        default=720,
        help='Output display height; use 0 to keep source height.',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Maximum frames to process; 0 means the complete video.',
    )
    parser.add_argument(
        '--warmup', type=int, default=2, help='Warmup forwards on the first frame.'
    )
    parser.add_argument(
        '--progress-every', type=int, default=30, help='Print progress every N frames.'
    )
    parser.add_argument(
        '--codec', default='mp4v', help='FourCC codec, default mp4v for broad OpenCV support.'
    )
    return parser.parse_args()


def make_title_bar(width: int, height: int, left: str, right: str) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        bar, left, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        bar, right, (width // 2 + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.line(bar, (width // 2, 0), (width // 2, height - 1), (255, 255, 255), 1)
    return bar


def make_side_by_side(
    original: np.ndarray,
    prediction: np.ndarray,
    display_height: int,
) -> np.ndarray:
    source_height, source_width = original.shape[:2]
    if display_height <= 0:
        display_height = source_height
    display_height = max(2, display_height - (display_height % 2))
    display_width = max(2, int(round(source_width * display_height / source_height)))
    display_width -= display_width % 2

    original = cv2.resize(
        original, (display_width, display_height), interpolation=cv2.INTER_AREA
    )
    prediction = cv2.resize(
        prediction, (display_width, display_height), interpolation=cv2.INTER_LINEAR
    )
    prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
    body = np.concatenate([original, prediction], axis=1)
    title = make_title_bar(
        body.shape[1], 36, 'Original video', 'Predicted moire edge map'
    )
    return np.concatenate([title, body], axis=0)


def main() -> None:
    args = parse_args()
    if args.size < 1:
        raise ValueError('--size must be >= 1.')
    if args.display_height < 0:
        raise ValueError('--display-height must be >= 0.')
    if args.max_frames < 0:
        raise ValueError('--max-frames must be >= 0.')
    if args.warmup < 0:
        raise ValueError('--warmup must be >= 0.')
    if args.progress_every < 1:
        raise ValueError('--progress-every must be >= 1.')
    if len(args.codec) != 4:
        raise ValueError('--codec must contain exactly four characters.')

    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint).to(device).eval()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f'OpenCV could not open the input video: {args.input}')
    if hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    frame_index = 0
    model_elapsed_ms: list[float] = []
    started_wall = time.perf_counter()
    first_frame_shape = None

    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if args.max_frames and frame_index >= args.max_frames:
                    break

                tensor = preprocess_image(frame, args.size, args.color_order).to(device)
                if frame_index == 0:
                    for _ in range(args.warmup):
                        model(tensor)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)

                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started_model = time.perf_counter()
                predictions, _ = model(tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                model_elapsed_ms.append((time.perf_counter() - started_model) * 1000.0)

                prediction = predictions[0][0, 0].float().cpu().numpy()
                prediction = np.clip(prediction, 0.0, 255.0).astype(np.uint8)
                comparison = make_side_by_side(frame, prediction, args.display_height)

                if writer is None:
                    first_frame_shape = list(comparison.shape)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*args.codec),
                        source_fps,
                        (comparison.shape[1], comparison.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f'OpenCV could not open output writer for {args.output}; '
                            f'try --codec avc1 or mp4v.'
                        )

                writer.write(comparison)
                frame_index += 1
                if frame_index % args.progress_every == 0:
                    total = f'/{reported_frames}' if reported_frames > 0 else ''
                    print(f'processed {frame_index}{total} frames', flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_index == 0:
        raise RuntimeError('The input video contained no readable frames.')

    wall_seconds = time.perf_counter() - started_wall
    metadata = {
        'input': str(args.input.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'output': str(args.output.resolve()),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'model_name': checkpoint['config']['arch']['model_name'],
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'color_order': args.color_order,
        'source_fps': source_fps,
        'source_size': [source_width, source_height],
        'reported_frames': reported_frames,
        'processed_frames': frame_index,
        'output_shape_hwc': first_frame_shape,
        'model_ms_mean': float(np.mean(model_elapsed_ms)),
        'model_ms_min': float(np.min(model_elapsed_ms)),
        'model_ms_max': float(np.max(model_elapsed_ms)),
        'wall_seconds': wall_seconds,
        'processing_fps': frame_index / wall_seconds,
        'audio_preserved': False,
        'codec': args.codec,
        'peak_gpu_memory_mib': (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == 'cuda'
            else None
        ),
        'torch_version': torch.__version__,
        'cuda_runtime': torch.version.cuda,
    }
    metadata_output = args.metadata_output or args.output.with_suffix('.json')
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata['metadata_output'] = str(metadata_output.resolve())
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
