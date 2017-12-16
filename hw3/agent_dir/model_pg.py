import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self,action_count):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.dense1 = nn.Linear(2048, 128)
        self.dense2 = nn.Linear(128, action_count)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.dense1(x.view(x.size(0),-1))
        x = F.relu(x)
        action_scores = self.dense2(x)
        return F.softmax(action_scores, dim=1)

