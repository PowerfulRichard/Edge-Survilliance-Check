import cv2
import time
import threading
import subprocess
import requests
import os
import uuid
import logging
import RPi.GPIO as GPIO
from concurrent.futures import ThreadPoolExecutor

# ================= 配置参数区 =================
RTSP_URL = "rtsp://admin:hzk0714...@mercury.hurl.live:554/stream2"
TARGET_FPS = 1             # 变量x：触发后每秒调取的帧数
RESIZE_WIDTH = 320         # 缩放宽度
RESIZE_HEIGHT = 240        # 缩放高度
SENSOR_PIN = 18            # 人体传感器 GPIO 引脚 (BCM编码)
HTTP_POST_URL = "http://server.hurl.live:9090/api/report"
FACE_APP_CMD = "./face_mini"
# ==============================================

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RTSPStreamReader:
    """
    RTSP 视频流后台读取类。
    保持持续抓取以清空底层缓冲区，确保主线程请求画面时，获取的永远是最新帧。
    """
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        self.lock = threading.Lock()
        
        # 启动后台守护线程
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                logging.warning("RTSP 流异常，尝试重新连接...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.src)

    def get_latest_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.cap.release()

def process_frame_task(frame):
    """
    具体的帧处理、识别及上报任务。交由线程池并发执行。
    """
    temp_img_path = ""
    try:
        # 1. 图像缩放
        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        
        # 2. 保存临时图片 (使用UUID确保多线程环境下文件相互独立，存入 /tmp 减少SD卡读写损耗)
        temp_img_path = f"/tmp/frame_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_img_path, resized_frame)
        
        # 3. 调用外部识别程序
        # subprocess.run 会阻塞当前工作线程，但不会阻塞主控抓帧循环
        process = subprocess.run(
            [FACE_APP_CMD, temp_img_path],
            capture_output=True,
            text=True,
            timeout=5  # 设置合理的超时时长，防止僵尸进程
        )
        
        output = process.stdout.strip()
        
        # 4. 解析输出结果
        name = "stranger"
        confidence = 0.0
        
        if output and output != "stranger":
            parts = output.split(',')
            if len(parts) >= 2:
                name = parts[0].strip()
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    logging.error(f"置信度数值解析错误: {parts[1]}")
        
        logging.info(f"识别完成 -> 结果: {name}, 置信度: {confidence}")
        
        # 5. HTTP 提交信息
        payload = {
            "name": name,
            "confidence": confidence,
            "timestamp": time.time()
        }
        # 发送 POST 请求，配置3秒超时防止网络波动卡死当前工作线程
        response = requests.post(HTTP_POST_URL, json=payload, timeout=3)
        logging.info(f"HTTP 上报完成 -> 状态码: {response.status_code}")

    except Exception as e:
        logging.error(f"任务执行期间发生异常: {e}")
    finally:
        # 6. 清理临时文件，维护存储空间健康
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

def main():
    # 初始化 GPIO
    GPIO.setmode(GPIO.BCM)
    # 假设传感器触发为高电平，平时需要下拉电阻维持低电平抗干扰
    GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    # 建立 RTSP 流读取实例
    logging.info("初始化 RTSP 流及缓存清理器...")
    stream = RTSPStreamReader(RTSP_URL)
    time.sleep(2)  # 等待缓冲区初始化完毕
    
    # 初始化线程池，max_workers 决定了可同时处理的最大识别任务数
    # 根据需求：2秒耗时 * x帧/秒。如果 x=2，则理论上并发任务维持在 4 个左右。
    worker_pool = ThreadPoolExecutor(max_workers=TARGET_FPS * 3)
    
    frame_interval = 1.0 / TARGET_FPS
    logging.info("系统初始化完毕，等待传感器高电平触发...")
    
    try:
        while True:
            # 判断 GPIO 状态是否为高电平
            if GPIO.input(SENSOR_PIN) == GPIO.HIGH:
                loop_start = time.time()
                
                # 获取无延迟的当前最新帧
                current_frame = stream.get_latest_frame()
                if current_frame is not None:
                    # 提交至线程池异步执行
                    worker_pool.submit(process_frame_task, current_frame)
                else:
                    logging.warning("尝试截取画面失败，未获取到有效帧数据。")
                
                # 动态计算休眠时间以维持稳定的采样帧率
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, frame_interval - elapsed)
                time.sleep(sleep_time)
            else:
                # 未触发状态下，适当休眠以释放 CPU 资源
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        logging.info("捕获到中断信号，正在终止运行环境...")
    finally:
        # 释放系统及硬件资源
        stream.stop()
        worker_pool.shutdown(wait=True)
        GPIO.cleanup()
        logging.info("资源已安全释放。")

if __name__ == "__main__":
    main()
