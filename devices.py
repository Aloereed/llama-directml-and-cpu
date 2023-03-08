import torch,torch_directml

def get_device():
    if torch.cuda.is_available():
        print("CUDA is used.")
        return torch.device("cuda")

    if torch_directml.is_available() and torch_directml.gpu_memory()[0]>=34359738368:
        print("DirectML GPU is used.")
        return torch_directml.device()
    print("CPU is used.")
    return torch.device("cpu")

device=get_device()