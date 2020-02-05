import torch

class converge():
    def __init__(self, condition=0.001):
        self.count = 0
        self.prev_loss = None
        self.condition = condition

    def check_converge(self, loss):
        if self.count > 5:
            return True
        # history exits
        if self.prev_loss is None:
            self.prev_loss = loss
        else:
            res = torch.abs(self.prev_loss - loss)
            if res < self.condition:
                self.count += 1
            else:
                self.count = 0
                self.prev_loss = None

        return False