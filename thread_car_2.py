import cv2
import time
import threading
import sys
import msvcrt  # Windows的键盘输入模块
from robomaster import robot
import math
import cv2
import tkinter as tk
import ctypes

class VideoThread(threading.Thread):
    def __init__(self, robot_instance):
        """
        视频采集显示线程
        :param robot_instance: 机器人实例
        """
        threading.Thread.__init__(self)
        self.robot = robot_instance
        self.camera = robot_instance.camera
        self.running = True
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.daemon = True  # 设置为守护线程，主线程结束时自动退出
        self.name = "VideoThread"

    def run(self):
        """视频采集和显示主循环"""
        print(f"[{self.name}] 开始采集视频流")

        # 初始化相机
        try:
            self.camera.start_video_stream(display=False, resolution='720p')
            print(f"[{self.name}] 相机初始化成功")
        except Exception as e:
            print(f"[{self.name}] 相机初始化失败: {e}")
            return

        # # 创建窗口
        # cv2.namedWindow('RoboMaster Camera', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('RoboMaster Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.resizeWindow('RoboMaster Camera', 640, 480)

        window_name = 'RoboMaster Camera'
        width, height = 800, 400

        # 1. 获取当前屏幕分辨率，计算居中的左上角坐标 (x, y)
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()  # 获取完信息后销毁临时实例

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # 2. 创建窗口 (WINDOW_GUI_NORMAL 可以隐藏 OpenCV 自带的顶部/底部工具栏)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, x, y)

        # 3. 强行去除操作系统边框 (此操作主要针对 Windows 环境)
        try:
            # 找到刚才创建的 OpenCV 窗口句柄
            HWND = ctypes.windll.user32.FindWindowW(None, window_name)
            if HWND:
                # 获取当前窗口的系统样式
                style = ctypes.windll.user32.GetWindowLongW(HWND, -16)
                # 通过位运算移除标题栏 (WS_CAPTION) 和 边框 (WS_THICKFRAME)
                style = style & ~0x00C00000 & ~0x00040000
                ctypes.windll.user32.SetWindowLongW(HWND, -16, style)
                # 强制刷新窗口以应用最新的无边框样式
                ctypes.windll.user32.SetWindowPos(HWND, 0, x, y, width, height, 0x0027)
        except Exception:
            pass # 如果你在 Mac 或 Ubuntu 下运行，会静默跳过硬去边框操作，保持普通的居中窗口

        while self.running:
            try:
                # 获取图像
                img = self.recv_image()

                if img is not None:
                    # 计算并显示FPS
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:  # 每30帧计算一次FPS
                        current_time = time.time()
                        self.fps = 30 / (current_time - self.fps_start_time)
                        self.fps_start_time = current_time

                    # 在图像上添加FPS信息
                    cv2.putText(img, f'FPS: {self.fps:.1f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 添加操作提示
                    cv2.putText(img, 'Controls: 1-6 Move | 0 Stop | q Quit', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 显示图像
                    cv2.imshow('RoboMaster Camera', img)

                    # 处理OpenCV窗口事件（必须调用，否则窗口无响应）
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(f"[{self.name}] 收到退出信号")
                        self.running = False
                        break

            except Exception as e:
                print(f'[{self.name}] 错误: {e}')
                time.sleep(0.1)

        # 清理资源
        cv2.destroyAllWindows()
        try:
            self.camera.stop_video_stream()
            print(f"[{self.name}] 相机已关闭")
        except:
            pass
        print(f"[{self.name}] 停止采集")

    def recv_image(self):
        """从相机接收图像"""
        try:
            # 获取最新帧
            img = self.camera.read_cv2_image(strategy='newest')
            return img
        except Exception as e:
            print(f"[{self.name}] 获取图像失败: {e}")
            return None

    def stop(self):
        """停止视频线程"""
        self.running = False


class IntegratedRobotController(threading.Thread):
    def __init__(self, robot_instance):
        super().__init__(name="RobotExecutor", daemon=True)
        self.robot = robot_instance
        self.chassis = robot_instance.chassis

        # 运行控制
        self.running = True
        self.latest_cmd = '0'  # 键盘实时输入的最新指令
        self.current_executed_cmd = None  # 状态机当前正在执行的指令
        self.last_input_time = time.time()

        # 运动参数
        self.speed = 0.3  # 线速度 m/s
        self.turn_speed = 15  # 角速度 deg/s
        self.timeout_limit = 0.5  # 如果0.5秒没有按键输入，自动停止 (模拟松手检测)

        # 键盘映射映射
        self.key_map = {
            'w': '1', '1': '1',  # 前进
            's': '2', '2': '2',  # 后退
            'a': '3', '3': '3',  # 左平移
            'd': '4', '4': '4',  # 右平移
            'q': '5', '5': '5',  # 左转
            'e': '6', '6': '6',  # 右转
            'z': '0', '0': '0'  # 停止
        }

    def get_key(self):
        """非阻塞获取键盘输入"""
        if msvcrt.kbhit():
            try:
                char = msvcrt.getch().decode('utf-8').lower()
                return char
            except:
                return None
        return None

    def _apply_movement(self, cmd):
        """底层硬件控制逻辑"""
        try:
            if cmd == '1':  # 前进
                self.chassis.drive_speed(x=self.speed, y=0, z=0)
            elif cmd == '2':  # 后退
                self.chassis.drive_speed(x=-self.speed, y=0, z=0)
            elif cmd == '3':  # 左移
                self.chassis.drive_speed(x=0, y=-self.speed, z=0)
            elif cmd == '4':  # 右移
                self.chassis.drive_speed(x=0, y=self.speed, z=0)
            elif cmd == '5':  # 左转
                self.chassis.drive_speed(x=0, y=0, z=-self.turn_speed)
            elif cmd == '6':  # 右转
                self.chassis.drive_speed(x=0, y=0, z=self.turn_speed)
            else:  # 停止 ('0' 或其他)
                self.chassis.drive_speed(x=0, y=0, z=0)
        except Exception as e:
            print(f"\n[Executor] 硬件执行异常: {e}")

    def run(self):
        """【线程】状态机执行器循环"""
        print(f"[{self.name}] 执行器线程已启动...")

        while self.running:
            now = time.time()
            active_cmd = self.latest_cmd

            # 自动复位逻辑：如果超过 timeout 没按键，设为停止状态
            if now - self.last_input_time > self.timeout_limit:
                active_cmd = '0'

            # 状态切换判断：只有指令变化时才操作硬件
            if active_cmd != self.current_executed_cmd:
                print(f"\r[{self.name}] 状态切换: {self.current_executed_cmd} -> {active_cmd}   ", end='')
                self._apply_movement(active_cmd)
                self.current_executed_cmd = active_cmd

            time.sleep(0.05)  # 20Hz 刷新率

    def start_control(self):
        """【主循环】键盘监听界面"""
        self.print_menu()
        self.start()  # 启动状态机执行线程

        try:
            while self.running:
                key = self.get_key()
                print(key)

                if key == 'z':  # 退出键
                    print("\n[User] 正在退出控制...")
                    self.running = False
                    break

                if key in self.key_map:
                    self.latest_cmd = self.key_map[key]
                    self.last_input_time = time.time()  # 更新输入时间

                time.sleep(0.02)  # 高频监听键盘

        finally:
            self.running = False
            self.chassis.drive_speed(x=0, y=0, z=0)  # 确保退出时停止

    def print_menu(self):
        print("\n" + "=" * 40)
        print("   机器人综合控制系统 (状态机模式)")
        print("-" * 40)
        print(" 控制方式: W/S/A/D 或 数字键 1-4")
        print(" 转向方式: Q/E     或 数字键 5-6")
        print(" 停止: 空格 或 0   | 退出: Z")
        print(" 模式: 连续触发 (按住保持，0.5s无输入停止)")
        print("=" * 40 + "\n")


def init_robot():
    """初始化机器人"""
    try:
        print("正在连接机器人...")
        robot_instance = robot.Robot()
        robot_instance.initialize(conn_type="ap")
        robot_instance.set_robot_mode(mode=robot.GIMBAL_LEAD)
        print("机器人连接成功！")
        time.sleep(1)
        return robot_instance

    except Exception as e:
        print(f"机器人连接失败: {e}")
        print("\n请检查:")
        print("1. 机器人是否已开机")
        print("2. 电脑是否已连接到机器人的WiFi")
        return None


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RoboMaster EP 双线程控制器")
    print("=" * 60)

    # 初始化机器人
    robot_instance = init_robot()
    if not robot_instance:
        sys.exit(1)

    # 创建线程实例
    video_thread = VideoThread(robot_instance)
    keyboard_controller = IntegratedRobotController(robot_instance)

    try:
        # 1. 启动视频线程 (后台)
        print(f"\n[Main] 启动视频线程...")
        video_thread.start()

        # 2. 启动键盘执行器线程 (后台状态机)
        print(f"[Main] 启动底层控制状态机...")
        keyboard_controller.start()

        # 3. 重要：主线程现在应该运行键盘监听循环
        # 而不是直接退出
        print("[Main] 进入主控制循环，按 Z 退出程序...")

        # 运行键盘监听循环（使用IntegratedRobotController中的逻辑）
        # 注意：这里直接使用keyboard_controller对象，但需要访问其方法
        keyboard_controller.print_menu()

        # 主线程的键盘监听循环
        while True:
            key = keyboard_controller.get_key()
            if key:
                print(f"按键: {key}")

            if key == 'z':  # 退出键
                print("\n[Main] 正在退出程序...")
                break

            if key in keyboard_controller.key_map:
                keyboard_controller.latest_cmd = keyboard_controller.key_map[key]
                keyboard_controller.last_input_time = time.time()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n[Main] 接收到 Ctrl+C，正在退出...")
    except Exception as e:
        print(f"\n[Main] 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n[Main] 正在清理资源...")

        # 停止标志位设置
        keyboard_controller.running = False
        video_thread.stop()

        # 等待线程结束
        video_thread.join(timeout=2.0)
        keyboard_controller.join(timeout=2.0)

        # 停止所有运动并关闭
        try:
            robot_instance.chassis.drive_speed(x=0, y=0, z=0)
            robot_instance.close()
            print("[Main] 机器人连接已关闭")
        except:
            pass

        print("[Main] 程序已退出")


if __name__ == '__main__':
    main()