#!/usr/bin/env python3
"""Generate the Waymo relighting 3x3 grid teaser video."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCENES = [3, 19, 22, 34, 49, 86, 114, 172, 703]
NIGHT_STAGE_OVERRIDES = {
    3: "waymo_1cam_3_relight_headlight_86_107.mp4",
}


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    frame_mode: str = "absolute"


STAGES = [
    Stage("input", "Input"),
    Stage("relight_daylight", "Daylight"),
    Stage("relight_evening", "Evening"),
    Stage("relight_headlight", "Night"),
    Stage("diverse_simulation", "Diverse Simulation", "local"),
]

CELL_WIDTH = 960
CELL_HEIGHT = 640
GRID_COLS = 3
GRID_ROWS = 3


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.next_frame = 0

    def read(self, frame_index: int) -> np.ndarray:
        if frame_index != self.next_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self.next_frame = frame_index

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_index} from {self.path}")

        self.next_frame += 1
        return normalize_frame(frame, self.path)

    def close(self) -> None:
        self.cap.release()


def normalize_frame(frame: np.ndarray, path: Path) -> np.ndarray:
    height, width = frame.shape[:2]
    if (width, height) == (CELL_WIDTH, CELL_HEIGHT):
        return frame
    resized = cv2.resize(frame, (CELL_WIDTH, CELL_HEIGHT), interpolation=cv2.INTER_AREA)
    print(f"Resized {path.name}: {width}x{height} -> {CELL_WIDTH}x{CELL_HEIGHT}")
    return resized


def stage_path(input_dir: Path, scene: int, stage: Stage) -> Path:
    if stage.key == "relight_headlight" and stage.label == "Night" and scene in NIGHT_STAGE_OVERRIDES:
        return input_dir / NIGHT_STAGE_OVERRIDES[scene]
    if stage.key == "diverse_simulation":
        matches = [
            path
            for path in sorted(input_dir.glob(f"waymo_1cam_{scene}_relight_headlight_*.mp4"))
            if path.name != NIGHT_STAGE_OVERRIDES.get(scene)
        ]
        if len(matches) != 1:
            match_list = "\n".join(str(match) for match in matches) or "none"
            raise FileNotFoundError(
                f"Expected exactly one diverse simulation video for scene {scene}; found:\n"
                f"{match_list}"
            )
        return matches[0]
    return input_dir / f"waymo_1cam_{scene}_{stage.key}.mp4"


def validate_inputs(input_dir: Path, start_frame: int, edges: list[int]) -> None:
    missing = []
    for scene in SCENES:
        for stage_index, stage in enumerate(STAGES):
            try:
                path = stage_path(input_dir, scene, stage)
            except FileNotFoundError as error:
                missing.append(str(error))
                continue

            if not path.exists():
                missing.append(str(path))
                continue

            cap = cv2.VideoCapture(str(path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            max_frame = stage_max_frame(stage, stage_index, scene, start_frame, edges)
            if frame_count <= max_frame:
                raise RuntimeError(
                    f"{path} has {frame_count} frames; frame {max_frame} is required"
                )
            if (width, height) != (CELL_WIDTH, CELL_HEIGHT):
                raise RuntimeError(
                    f"{path} is {width}x{height}; expected {CELL_WIDTH}x{CELL_HEIGHT}"
                )

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required videos:\n{joined}")


def stage_source_frame(
    stage: Stage,
    stage_start: int,
    scene: int,
    start_frame: int,
    output_frame: int,
) -> int:
    if stage.key == "relight_headlight" and stage.label == "Night" and scene in NIGHT_STAGE_OVERRIDES:
        return output_frame - stage_start
    if stage.frame_mode == "local":
        return output_frame - stage_start
    return start_frame + output_frame


def stage_max_frame(
    stage: Stage,
    stage_index: int,
    scene: int,
    start_frame: int,
    edges: list[int],
) -> int:
    stage_start = edges[stage_index]
    stage_end = edges[stage_index + 1] - 1
    return stage_source_frame(stage, stage_start, scene, start_frame, stage_end)


def chunk_edges(total_frames: int, stage_count: int) -> list[int]:
    return np.linspace(0, total_frames, stage_count + 1, dtype=int).tolist()


def stage_for_frame(output_frame: int, edges: list[int]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= output_frame < edges[index + 1]:
            return index
    return len(edges) - 2


def make_grid(
    readers: dict[tuple[int, int], VideoReader],
    stage_index: int,
    stage_start: int,
    start_frame: int,
    output_frame: int,
) -> np.ndarray:
    cells = []
    stage = STAGES[stage_index]
    for scene in SCENES:
        source_frame = stage_source_frame(stage, stage_start, scene, start_frame, output_frame)
        cells.append(readers[(scene, stage_index)].read(source_frame))

    rows = []
    for row in range(GRID_ROWS):
        first = row * GRID_COLS
        rows.append(np.hstack(cells[first : first + GRID_COLS]))
    return np.vstack(rows)


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.15
    thickness = 3
    margin_x = 30
    margin_y = 28
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size
    box_width = text_width + margin_x * 2
    box_height = text_height + margin_y * 2

    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)
    cv2.putText(
        output,
        label,
        (margin_x, margin_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def diagonal_wipe(old_frame: np.ndarray, new_frame: np.ndarray, step: int, steps: int) -> np.ndarray:
    height, width = old_frame.shape[:2]
    progress = (step + 1) / steps
    yy, xx = np.mgrid[0:height, 0:width]

    diagonal_span = int(height * 0.45)
    line_x = -diagonal_span + progress * (width + diagonal_span * 2) + (height - yy) * 0.32
    distance = line_x - xx

    band = 42.0
    alpha = np.clip((distance + band) / (band * 2), 0.0, 1.0).astype(np.float32)
    alpha = alpha[..., None]
    blended = new_frame.astype(np.float32) * alpha + old_frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def open_readers(input_dir: Path) -> dict[tuple[int, int], VideoReader]:
    readers = {}
    for scene in SCENES:
        for stage_index, stage in enumerate(STAGES):
            readers[(scene, stage_index)] = VideoReader(stage_path(input_dir, scene, stage))
    return readers


def close_readers(readers: dict[tuple[int, int], VideoReader]) -> None:
    for reader in readers.values():
        reader.close()


def encode_h264(raw_path: Path, output_path: Path, crf: int, preset: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(raw_path),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def generate(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    total_frames = args.end_frame - args.start_frame + 1
    edges = chunk_edges(total_frames, len(STAGES))
    grid_size = (CELL_WIDTH * GRID_COLS, CELL_HEIGHT * GRID_ROWS)

    validate_inputs(input_dir, args.start_frame, edges)
    readers = open_readers(input_dir)

    temp_context = tempfile.TemporaryDirectory(prefix="waymo_relighting_grid_")
    temp_dir = Path(temp_context.name)
    if args.keep_temp:
        temp_context.cleanup()
        temp_dir = args.keep_temp.resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)

    raw_path = temp_dir / "waymo_relighting_grid_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        grid_size,
    )
    if not writer.isOpened():
        close_readers(readers)
        raise RuntimeError(f"Failed to open video writer: {raw_path}")

    frozen_grids: dict[tuple[int, int], np.ndarray] = {}
    try:
        for output_frame in range(total_frames):
            current_stage = stage_for_frame(output_frame, edges)
            stage_start = edges[current_stage]
            transition_step = output_frame - stage_start

            in_transition = current_stage > 0 and transition_step < args.transition_frames
            if in_transition:
                old_stage = current_stage - 1
                old_stage_start = edges[old_stage]
                old_key = (old_stage, stage_start - 1)
                new_key = (current_stage, output_frame)
                if old_key not in frozen_grids:
                    frozen_grids[old_key] = make_grid(
                        readers,
                        old_stage,
                        old_stage_start,
                        args.start_frame,
                        stage_start - 1,
                    )
                if new_key not in frozen_grids:
                    frozen_grids[new_key] = make_grid(
                        readers,
                        current_stage,
                        stage_start,
                        args.start_frame,
                        output_frame,
                    )
                grid = diagonal_wipe(
                    frozen_grids[old_key],
                    frozen_grids[new_key],
                    transition_step,
                    args.transition_frames,
                )
            else:
                grid = make_grid(
                    readers,
                    current_stage,
                    stage_start,
                    args.start_frame,
                    output_frame,
                )

            frame = draw_label(grid, STAGES[current_stage].label)
            writer.write(frame)
            print(f"Wrote frame {output_frame + 1}/{total_frames}", end="\r", flush=True)
    finally:
        writer.release()
        close_readers(readers)

    print()
    encode_h264(raw_path, output_path, args.crf, args.preset)
    print(f"Wrote {output_path}")
    if not args.keep_temp:
        temp_context.cleanup()
    else:
        print(f"Kept temporary files in {temp_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("static/videos/all_relight"))
    parser.add_argument("--output", type=Path, default=Path("static/videos/waymo_relighting_grid.mp4"))
    parser.add_argument("--start-frame", type=int, default=20)
    parser.add_argument("--end-frame", type=int, default=130)
    parser.add_argument("--transition-frames", type=int, default=3)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--keep-temp", type=Path)
    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.end_frame < args.start_frame:
        parser.error("--end-frame must be greater than or equal to --start-frame")
    if args.transition_frames < 0:
        parser.error("--transition-frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


if __name__ == "__main__":
    generate(parse_args())
