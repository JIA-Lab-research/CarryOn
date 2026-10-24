"""Write MP4 debug videos that VS Code and browsers can play (H.264 + yuv420p).

OpenCV's ``mp4v`` (MPEG-4 Part 2) is often not previewable in VS Code; we encode
with libx264 via imageio/ffmpeg instead.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# VS Code / QuickTime / browsers expect AVC in MP4 with yuv420p chroma subsampling.
H264_OUTPUT_PARAMS = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def resolve_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def write_mp4(
    path: pathlib.Path | str,
    frames: Sequence[np.ndarray],
    *,
    fps: float = 10.0,
) -> None:
    """Encode RGB uint8 frames to an H.264 MP4."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        log.warning("write_mp4: no frames for %s", path)
        return

    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path),
        fps=max(1e-3, float(fps)),
        codec="libx264",
        output_params=H264_OUTPUT_PARAMS,
        macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


class StreamingMp4Writer:
    """Incremental H.264 writer; opens on the first appended frame."""

    def __init__(self, path: pathlib.Path | str, *, fps: float = 10.0) -> None:
        self.path = pathlib.Path(path)
        self.fps = max(1e-3, float(fps))
        self._writer = None
        self.frames = 0

    def append(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(
                str(self.path),
                fps=self.fps,
                codec="libx264",
                output_params=H264_OUTPUT_PARAMS,
                macro_block_size=1,
            )
        self._writer.append_data(rgb)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
