import socket


def tcp_server():
    # 创建socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 设置端口重用（可选，避免地址被占用）
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定地址和端口
    host = '0.0.0.0'  # 监听所有网络接口
    port = 8811
    server_socket.bind((host, port))

    # 开始监听，最大等待连接数5
    server_socket.listen(5)
    print(f"服务器启动，监听 {host}:{port}")

    try:
        while True:
            # 接受客户端连接
            print("等待客户端连接...")
            client_socket, client_address = server_socket.accept()
            print(f"收到连接: {client_address}")

            try:
                # 接收数据
                data = client_socket.recv(1024)
                print(data)
                if data:
                    print(f"收到数据: {data.decode('utf-8')}")

                    # 发送响应
                    response = f"服务器已收到: {data.decode('utf-8')}"
                    client_socket.send(response.encode('utf-8'))
                    print(f"已发送响应: {response}")
                else:
                    print("客户端已断开连接")

            except socket.error as e:
                print(f"数据传输错误: {e}")
            finally:
                # 关闭客户端连接
                client_socket.close()
                print(f"已关闭连接: {client_address}")

    except KeyboardInterrupt:
        print("\n服务器正在关闭...")
    finally:
        server_socket.close()
        print("服务器已关闭")


if __name__ == "__main__":
    tcp_server()