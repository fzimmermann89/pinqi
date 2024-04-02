import numpy as np
import torch


def get_emptiest_gpu():
    results = []
    gpus = torch.cuda.device_count()
    if gpus == 1:
        return 0
    try:
        import nvidia_smi

        nvidia_smi.nvmlInit()
    except:
        return 0
    for gpuid in range(gpus):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        results.append(info.free)
    nvidia_smi.nvmlShutdown()
    best = int(np.argmax(results))
    return best


def use_emptiest_gpu():
    torch.cuda.set_device(get_emptiest_gpu())