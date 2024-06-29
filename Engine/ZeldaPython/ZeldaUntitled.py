import os
import json
import socket
import random
import math

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
    "ProfabName": "",
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
        "Position": [5.0, 5.0, 5.0],
        "Lookat": [0.0, 0.0, 0.5],
        "Speed": 2.5,
        "FOV": 45.0,
        "zNear": 0.1,
        "zFar": 45.0
    },
    "Skydome": {
        "EnableSkydome": True,
        "OverrideSkydome": True,
        "SkydomeFileName": "grassland_night.png",
        "OverrideCubemap": True,
        "CubemapFileNames": [
            "grassland_night_X0.png",
            "grassland_night_X1.png",
            "grassland_night_Y2.png",
            "grassland_night_Y3.png",
            "grassland_night_Z4.png",
            "grassland_night_Z5.png"
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

terrain = xkObject.copy()
terrain["RenderFlags"] = 0
terrain["ProfabName"] = "terrain"
terrain["InstanceCount"] = 1
xkWorld["Objects"].append(terrain)

rock_01 = xkObject.copy()
rock_01["RenderFlags"] = 0
rock_01["ProfabName"] = "rock_01"
rock_01["InstanceCount"] = 1
xkWorld["Objects"].append(rock_01)

rock_02 = xkObject.copy()
rock_02["RenderFlags"] = 0
rock_02["ProfabName"] = "rock_02"
rock_02["InstanceCount"] = 64
rock_02["MinRadius"] = 1.0
rock_02["MaxRadius"] = 5.0
rock_02["MinPScale"] = 0.2
rock_02["MaxPScale"] = 0.5
xkWorld["Objects"].append(rock_02)

grass_01 = xkObject.copy()
grass_01["RenderFlags"] = 0
grass_01["ProfabName"] = "grass_01"
grass_01["InstanceCount"] = 10000
grass_01["MinRadius"] = 2.0
grass_01["MaxRadius"] = 8.0
grass_01["MinPScale"] = 0.1
grass_01["MaxPScale"] = 0.5
xkWorld["Objects"].append(grass_01)

grass_02 = xkObject.copy()
grass_02["RenderFlags"] = 0
grass_02["ProfabName"] = "grass_02"
grass_02["InstanceCount"] = 10000
grass_02["MinRadius"] = 1.0
grass_02["MaxRadius"] = 9.0
grass_02["MinPScale"] = 0.1
grass_02["MaxPScale"] = 0.5
xkWorld["Objects"].append(grass_02)

Moonlight = xkLight.copy()
Moonlight["Position"] = [20.0, 0.0, 20.0]
Moonlight["Type"] = 0
Moonlight["Color"] = [0.0, 0.1, 0.6]
Moonlight["Intensity"] = 15.0
Moonlight["Direction"] = Moonlight["Position"]
Moonlight["Radius"] = 0.0
Moonlight["ExtraData"] = [0.0, 0.0, 0.0, 0.0]
xkWorld["DirectionalLights"].append(Moonlight)

PointLightNum = 16
for i in range(PointLightNum):
    PointLight = xkLight.copy()
    random.seed(i)
    radians = random.uniform(0.0, 360.0)
    distance = random.uniform(0.1, 0.6)
    X = math.sin(math.radians(radians)) * distance
    Y = math.cos(math.radians(radians)) * distance
    Z = 1.0
    PointLight["Position"] = [X, Y, Z]
    PointLight["Type"] = 1
    R = random.uniform(0.5, 0.75)
    G = random.uniform(0.25, 0.5)
    B = 0.0
    PointLight["Color"] = [R, G, B]
    PointLight["Intensity"] = 10.0
    PointLight["Direction"] = [0.0, 0.0, 1.0]
    PointLight["Radius"] = 1.5
    PointLight["ExtraData"] = [0.0, 0.0, 0.0, 0.0]
    xkWorld["PointLights"].append(PointLight)


# send data to engine
# json_str = json.dumps(xkWorld)
# sendDataToEngine(json_str)

