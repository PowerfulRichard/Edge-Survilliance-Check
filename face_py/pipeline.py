import cv2
import time
import threading
import subprocess
import requests
from requests.exceptions import ReadTimeout, ConnectTimeout
import os
import uuid
import logging
import RPi.GPIO as GPIO
from concurrent.futures import ThreadPoolExecutor

# ================= 配置参数区 =================
RTSP_URL = "rtsp://admin:hzk0714...@mercury.hurl.live:554/stream2"
TARGET_FPS = 1             # 触发后每秒调取的帧数 (变量x)
RESIZE_WIDTH = 320         # 缩放宽度
RESIZE_HEIGHT = 240        # 缩放高度
SENSOR_PIN = 18            # 人体传感器 GPIO 引脚 (BCM编码)
HTTP_POST_URL = "http://server.hurl.live:9090/api/report"
FACE_APP_CMD = "./face_mini"

# ================= 业务策略区 =================
# 上报冷却时间策略 (单位: 秒)
COOLDOWN_STRANGER = 10.0   # 连续未识别到人时，10秒内不重复触发上报
COOLDOWN_KNOWN = 3.0       # 同一个已知人员，3秒内不重复触发上报
# ==============================================

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportFilter:
    """
    状态机：用于解耦“画面识别频率”与“HTTP上报频率”。
    通过时间窗口机制实现防抖 (Debounce) 与冗余数据过滤。
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.last_name = None
        self.last_time = 0.0

    def should_report(self, name):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time

            if name == "stranger":
                # 过滤连续的 stranger 结果
                if self.last_name == "stranger" and elapsed < COOLDOWN_STRANGER:
                    return False
            else:
                # 过滤连续的相同已知人员结果
                if name == self.last_name and elapsed < COOLDOWN_KNOWN:
                    return False
            
            # 更新状态机并放行
            self.last_name = name
            self.last_time = now
            return True

# 实例化全局上报过滤器
report_filter = ReportFilter()

class RTSPStreamReader:
    # RTSP 视频流后台读取类，保持无间断抓取以清空底层缓冲区
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        self.lock = threading.Lock()
        
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
    # 并发执行的图像处理与识别任务 
    temp_img_path = ""
    try:
        # 1. 缩放与保存
        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        temp_img_path = f"/tmp/frame_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_img_path, resized_frame)
        
        # 2. 调用外部识别程序 (实际耗时 300-700ms)
        process = subprocess.run(
            #[FACE_APP_CMD, "recognize", temp_img_path],
            [FACE_APP_CMD, temp_img_path],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        output = process.stdout.strip()
        
        # 3. 解析结果
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
        
        # 4. 经过状态机判断是否需要触发 HTTP 提交
        if not report_filter.should_report(name):
            logging.debug(f"识别完成 -> {name} (受限于冷却策略，已被丢弃)")
            return
            
        logging.info(f"触发网络上报 -> 结果: {name}, 置信度: {confidence}")
        
        # 5. HTTP 提交 (解决 Read Timeout 问题)
        payload = {
            "name": name,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        try:
            # 关键修改：将 timeout 拆分为 (连接超时, 读取超时)
            # 服务端在连接后处理较慢，赋予 5 秒等待时间
            response = requests.post(HTTP_POST_URL, json=payload, timeout=(2.0, 5.0))
            logging.info(f"上报完成 -> 状态码: {response.status_code}")
        except ReadTimeout:
            # 服务端已收到请求但在 5 秒内未返回响应，视为非致命错误
            logging.warning("上报警告 -> 接收端处理缓慢引发 Read Timeout，但不影响指令实际送达。")
        except ConnectTimeout:
            logging.error("上报错误 -> 网络连接超时，无法触达服务器。")
        except Exception as req_e:
            logging.error(f"上报网络异常 -> {req_e}")

    except Exception as e:
        logging.error(f"任务执行异常: {e}")
    finally:
        # 6. 清理临时文件
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    logging.info("初始化 RTSP 流及缓存清理器...")
    stream = RTSPStreamReader(RTSP_URL)
    time.sleep(2)
    
    # 线程池配置：根据300-700ms耗时，适度降低并发数避免CPU争抢
    worker_pool = ThreadPoolExecutor(max_workers=TARGET_FPS * 2)
    frame_interval = 1.0 / TARGET_FPS
    
    logging.info("系统初始化完毕，等待传感器信号...")
    
    try:
        while True:
            if GPIO.input(SENSOR_PIN) == GPIO.HIGH:
                loop_start = time.time()
                
                current_frame = stream.get_latest_frame()
                if current_frame is not None:
                    worker_pool.submit(process_frame_task, current_frame)
                
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, frame_interval - elapsed)
                time.sleep(sleep_time)
            else:
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        logging.info("捕获到中断信号，正在终止运行...")
    finally:
        stream.stop()
        worker_pool.shutdown(wait=True)
        GPIO.cleanup()
        logging.info("资源释放完毕。")

if __name__ == "__main__":
    main()
