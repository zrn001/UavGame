import socket
import struct
import time

p_labels_cca = [1]  # 示例数据  

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 设置端口重用（可选，避免地址被占用）
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 绑定地址和端口
host = '0.0.0.0'  # 监听所有网络接口
port = 8811
server_socket.bind((host, port))

data = struct.pack('!i', p_labels_cca[0])
time.sleep(5)  # 确保服务器已准备好接收数据
server_socket.send(data)  # 确保所有数据都发送
print(f"已发送: {data}")
server_socket.close()



