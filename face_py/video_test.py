#!/usr/bin/env python3
"""
rtsp_face_mini.py
从 RTSP 流中每秒抓取 x 帧，缩放至 w*h，并调用 ./face_mini 程序处理
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="从 RTSP 流抓帧并调用 face_mini 处理"
    )
    parser.add_argument("rtsp_url", help="RTSP 流地址")
    parser.add_argument(
        "-x", "--fps", type=float, default=1.0, help="每秒抓取帧数，默认 1"
    )
    parser.add_argument(
        "-W", "--width", type=int, default=320, help="缩放后的宽度，默认 640"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=240, help="缩放后的高度，默认 480"
    )
    parser.add_argument(
        "--face-mini",
        default="./face_mini",
        help="face_mini 程序路径，默认 ./face_mini",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留临时图片文件（默认会删除）",
    )
    args = parser.parse_args()

    # 打开 RTSP 流
    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"无法打开 RTSP 流: {args.rtsp_url}", file=sys.stderr)
        sys.exit(1)

    # 获取原始帧率，用于计算抓帧间隔
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25  # 假设默认帧率
    print(f"原始视频帧率: {original_fps:.2f}")

    # 计算每隔多少原始帧抓取一帧，保证每秒抓取 args.fps 帧
    if args.fps <= 0:
        print("抓取帧率必须大于0", file=sys.stderr)
        sys.exit(1)
    frame_interval = max(1, int(original_fps / args.fps))
    print(f"每 {frame_interval} 帧抓取一帧，期望每秒抓取 {original_fps/frame_interval:.2f} 帧")

    frame_count = 0
    temp_dir = Path(tempfile.mkdtemp(prefix="rtsp_face_mini_"))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("流结束或读取失败，退出。")
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            # 缩放
            resized = cv2.resize(frame, (args.width, args.height))

            # 保存为临时图片
            temp_path = temp_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(temp_path), resized)

            # 调用 face_mini
            cmd = [args.face_mini, str(temp_path)]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,  # 防止卡死
                )
                print(f"[{temp_path.name}] 输出:\n{result.stdout}")
                if result.stderr:
                    print(f"[{temp_path.name}] 错误:\n{result.stderr}", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f"[{temp_path.name}] face_mini 执行超时", file=sys.stderr)
            except FileNotFoundError:
                print(f"找不到程序: {args.face_mini}", file=sys.stderr)
                sys.exit(1)

            # 如果不保留临时文件，立即删除
            if not args.keep_temp:
                temp_path.unlink(missing_ok=True)

    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    finally:
        cap.release()
        # 清理临时目录（如果还有文件）
        if not args.keep_temp:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"临时文件保存在: {temp_dir}")


if __name__ == "__main__":
    main()