import socket
from struct import pack, unpack
import threading

_send_lock = threading.Lock()

# 创建客户端socket
hostport = ('127.0.0.1', 30000)
cmd_socket = socket.create_connection(hostport, timeout=15)
cmd_socket.settimeout(None)

# 准备要发送的数组
array_to_send = [1.23, 4.56, 7.89]

# 将数组转换为字节流
x = 1
while True:
    x+=1
    array_to_send[0] = array_to_send[0]+x
    data = pack('3d', *array_to_send)
    with _send_lock:
        # 发送数据
        cmd_socket.send(data)  # 发送数据到套接字
        server_reply = cmd_socket.recv(24)  # 接收套接字数据
    temp_list = list(unpack("3d", server_reply))
    print(temp_list)
# 关闭连接
cmd_socket.close()
