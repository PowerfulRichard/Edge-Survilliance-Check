#!/usr/bin/env python3
"""Thin Python controller for the NCNN C++ executables."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    det_param: str
    det_bin: str
    rec_param: str
    rec_bin: str
    det_size: int = 320
    db: str = "db/faces.db"
    threshold: float = 0.45
    det_score: float = 0.45
    nms: float = 0.40
    threads: int = 4


class FaceCpp:
    def __init__(self, executable: str, config: ModelConfig):
        self.executable = str(Path(executable))
        self.config = config

    def _base_command(self, image: str) -> list[str]:
        c = self.config
        return [
            self.executable,
            "--image", image,
            "--det-param", c.det_param,
            "--det-bin", c.det_bin,
            "--rec-param", c.rec_param,
            "--rec-bin", c.rec_bin,
            "--db", c.db,
            "--det-size", str(c.det_size),
            "--threshold", str(c.threshold),
            "--det-score", str(c.det_score),
            "--nms", str(c.nms),
            "--threads", str(c.threads),
        ]

    def _run(self, cmd: list[str]) -> Dict[str, Any]:
        p = subprocess.run(cmd, text=True, capture_output=True)
        stdout = p.stdout.strip()
        if not stdout:
            raise RuntimeError(f"C++ produced no JSON. returncode={p.returncode}, stderr={p.stderr.strip()}")
        try:
            data = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid C++ JSON: {stdout}\nstderr: {p.stderr.strip()}") from exc
        data["cpp_returncode"] = p.returncode
        if p.stderr.strip():
            data["cpp_stderr"] = p.stderr.strip()
        return data

    def infer(self, image: str) -> Dict[str, Any]:
        cmd = self._base_command(image)
        # face_mini does not accept --mode; face_main/face_debug do.
        if Path(self.executable).name != "face_mini":
            cmd += ["--mode", "infer"]
        return self._run(cmd)

    def enroll(self, image: str, name: str) -> Dict[str, Any]:
        if Path(self.executable).name == "face_mini":
            raise ValueError("face_mini is inference-only; use face_main or face_debug for enrollment")
        cmd = self._base_command(image) + ["--mode", "enroll", "--name", name]
        return self._run(cmd)
