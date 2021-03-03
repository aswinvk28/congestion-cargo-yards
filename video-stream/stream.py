from .constants import RTSP_URL
import socket
import rtsp

HOST = "video_ui"               # Symbolic name meaning all available interfaces
PORT = 3001              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_TCP, socket.IPPROTO_IP)
bind_address = (HOST, PORT)
s.bind(bind_address) # bind to the agent
s.listen(1) # listen to any incoming requests

connection, address = s.accept()

with rtsp.Client(rtsp_server_uri = RTSP_URL) as client:

    while True:
        data = connection.recv(1024)

        image = client.read(raw=True)
        if not image:
            break
        connection.send(image)

