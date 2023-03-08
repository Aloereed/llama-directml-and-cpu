import torch,torch_directml

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch_directml.is_available() and torch_directml.gpu_memory()[0]>=34359738368:
        return torch_directml.device()

    return torch.device("cpu")

device=get_device()