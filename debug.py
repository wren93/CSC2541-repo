import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )

    def forward(self, x):
        return self.net(x)

model = Model()
model.cuda()
sd = model.cpu().state_dict()
print(type(model))