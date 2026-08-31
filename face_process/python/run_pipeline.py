#!/usr/bin/env python3
"""Example top-level Python flow: Python chooses models/images and calls C++."""
from __future__ import annotations

import argparse
import json
from face_cpp import FaceCpp, ModelConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpp", default="./build/face_main")
    p.add_argument("--mode", choices=["infer", "enroll"], default="infer")
    p.add_argument("--image", required=True)
    p.add_argument("--name")
    p.add_argument("--det-param", required=True)
    p.add_argument("--det-bin", required=True)
    p.add_argument("--rec-param", required=True)
    p.add_argument("--rec-bin", required=True)
    p.add_argument("--db", default="db/faces.db")
    p.add_argument("--det-size", type=int, choices=[160, 320, 480, 640], default=320)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--det-score", type=float, default=0.45)
    p.add_argument("--nms", type=float, default=0.40)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    cfg = ModelConfig(
        det_param=args.det_param,
        det_bin=args.det_bin,
        rec_param=args.rec_param,
        rec_bin=args.rec_bin,
        det_size=args.det_size,
        db=args.db,
        threshold=args.threshold,
        det_score=args.det_score,
        nms=args.nms,
        threads=args.threads,
    )
    engine = FaceCpp(args.cpp, cfg)

    if args.mode == "infer":
        result = engine.infer(args.image)
    else:
        if not args.name:
            p.error("--name is required in enroll mode")
        result = engine.enroll(args.image, args.name)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
