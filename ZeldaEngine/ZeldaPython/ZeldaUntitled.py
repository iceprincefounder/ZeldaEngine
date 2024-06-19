import os
import json
import socket

# @TODO: bind DLL failed! Try PyBind11
# from ctypes import cdll
# lib = cdll.LoadLibrary('D:/ZeldaEngine/build/Debug/ZeldaPythonLib.dll')
# lib.hello_world()

def sendDataToEngine(data, port=8080):
    try:
        # 创建一个socket对象
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 连接到给定端口和localhost
            s.connect(('localhost', port))
            # 发送数据
            s.sendall(data.encode())
            # 接收响应（可选）
            response = s.recv(1024)
            print('Received:', response.decode())
    except ConnectionRefusedError:
        print(f"Connection to port {port} failed. Make sure there's a server listening on this port.")
    except Exception as e:
        print(f"An error occurred: {e}")

xkLight = {
    "Position": [20.0, 0.0, 20.0],
    "Type": 0,
    "Color": [1.0, 1.0, 1.0],
    "Intensity": 3.0,
    "Direction": [0.7, 0.7, 0.7],
    "Radius": 0.0,
    "ExtraData": [0.0, 0.0, 0.0, 0.0]
}

xkObject = {
    "RenderFlags": 0,
    "ProfabName": "sword_and_shield",
    "InstanceCount": 1,
    "MinRadius": 0.0,
    "MaxRadius": 0.0,
    "MinRotYaw": 0.0,
    "MaxRotYaw": 0.0,
    "MinRotRoll": 0.0,
    "MaxRotRoll": 0.0,
    "MinRotPitch": 0.0,
    "MaxRotPitch": 0.0,
    "MinPScale": 0.0,
    "MaxPScale": 0.0
}

xkWorld = {
    "MainCamera": {
        "Position": [2.5, 2.5, 2.5],
        "Lookat": [0.0, 0.0, 0.0],
        "Speed": 2.5,
        "FOV": 45.0,
        "zNear": 0.1,
        "zFar": 45.0
    },
    "Skydome": {
        "EnableSkydome": False,
        "OverrideSkydome": True,
        "SkydomeFileName": "Content/Textures/skydome.png",
        "OverrideCubemap": True,
        "CubemapFileNames": [
            "Content/Textures/cubemap_X0.png",
            "Content/Textures/cubemap_X0.png",
            "Content/Textures/cubemap_X0.png",
            "Content/Textures/cubemap_X0.png",
            "Content/Textures/cubemap_X0.png",
            "Content/Textures/cubemap_X0.png"
        ]
    },
    "Background": {
        "EnableBackground": True,
        "OverrideBackground": True,
        "BackgroundFileName": "background.png"
    },
    "DirectionalLights": [],
    "PointLights":[],
    "SpotLights": [],
    "Objects": []
}

xkWorld["DirectionalLights"].append(xkLight)
xkWorld["Objects"].append(xkObject)

# 使用函数发送数据
# json_str = json.dumps(xkWorld)
# sendDataToEngine(json_str)