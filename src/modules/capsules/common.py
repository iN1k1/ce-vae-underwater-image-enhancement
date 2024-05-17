import torch


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = ((squared_norm / (1. + squared_norm)) /
                     torch.sqrt(squared_norm + torch.finfo(torch.float32).eps)) * input_tensor
    return output_tensor

