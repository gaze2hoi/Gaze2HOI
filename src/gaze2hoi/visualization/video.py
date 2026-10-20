"""Stream rendered RGB frames to an H.264 video without intermediate image files."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def validate_video_size(size: int) -> None:
    if type(size) is not int or size < 128 or size % 2:
        raise ValueError("Video size must be an even integer of at least 128 pixels")


def require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for video export; install it and add it to PATH")
    return executable


def encode_video(frames: Iterable[Any], output: str | Path, size: int, fps: int = 30) -> Path:
    validate_video_size(size)
    if type(fps) is not int or fps < 1:
        raise ValueError("Video frame rate must be a positive integer")
    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension")
    executable = require_ffmpeg()
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent
    )
    os.close(descriptor)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size}x{size}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-movflags",
        "+faststart",
        "-threads",
        "2",
        temporary,
    ]
    first = None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in frames:
                rgb = np.asarray(frame, dtype=np.uint8)
                if rgb.shape != (size, size, 3):
                    raise ValueError(
                        f"Expected RGB frame shape {(size, size, 3)}, received {rgb.shape}"
                    )
                if first is None:
                    first = frame.copy()
                process.stdin.write(rgb.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(
                    f"ffmpeg could not encode the video: {errors.read().decode(errors='replace')}"
                )
            if first is None:
                raise ValueError("Cannot encode an empty frame sequence")
            Path(temporary).replace(output)
            first.save(output.with_suffix(".webp"), lossless=True, method=6)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            Path(temporary).unlink(missing_ok=True)
    return output
