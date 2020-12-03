import torch

def micro_contrastive(left_left, left_right, right_right, right_left):
    count = (left_left.numel() + left_right.numel() + right_right.numel() + right_left.numel()) / 2
    sum = torch.sum(left_right) + torch.sum(right_left) - torch.sum(left_left) - torch.sum(right_right)
    return sum / count



def macro_contrastive(left_left, left_right, right_right, right_left):
    return (torch.mean(left_right)-torch.mean(left_left) + torch.mean(right_left)- torch.mean(right_right)) / 2


