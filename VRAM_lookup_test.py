from pynvml import *

nvmlInit()
print(F"Driver version: {nvmlSystemGetDriverVersion()}")
device_count = nvmlDeviceGetCount()
print(device_count)
handle = nvmlDeviceGetHandleByIndex(0)
print(handle)
print(f"Device Name: {nvmlDeviceGetName(handle)}")
info = nvmlDeviceGetMemoryInfo(handle)
print("Total Memory in MB: ", info.total/1024**2)
print("Free Memory in MB: ", info.free/1024**2)
print("Used Memory in MB: ", info.used/1024**2)

nvmlShutdown()