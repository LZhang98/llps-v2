import torch

class FixedLengthClassifier (torch.nn.Module):
    def __init__(self, seq_len) -> None:
        super().__init__()
        layer1 = torch.nn.Linear(seq_len, 240)
    
    def forward(self, x):
        x = self.layer1(1)

'''
        self.dense = torch.nn.Sequential(
                        torch.nn.Linear(171008, 1280),
                        torch.nn.ReLU(),
                        torch.nn.Linear(1280, 240),
                        torch.nn.ReLU(),
                        torch.nn.Linear(240, 48),
                        torch.nn.ReLU(),
                        torch.nn.Linear(48, 1),
                        torch.nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.pool_flatten(x)
        x = self.dense(x)
        return x
'''

class AdaptiveClassifier (torch.nn.Module):
    def __init__(self, model_dim) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.dense = torch.nn.Sequential(
                        torch.nn.Linear(model_dim, 48),
                        torch.nn.ReLU(),
                        torch.nn.Linear(48, 1),
                        torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool(x).permute(0,2,1).squeeze()
        x = self.dense(x)
        return x

'''
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        #move fc layers out in the future
        # self.fc1 = torch.nn.Linear(1280, 240)
        # self.fc2 = torch.nn.Linear(240, 48)
        # self.final = torch.nn.Linear(48, 1)
        self.dense = torch.nn.Sequential(
                        torch.nn.Linear(model_dim, 240),
                        torch.nn.ReLU(),
                        torch.nn.Linear(240, 48),
                        torch.nn.ReLU(),
                        torch.nn.Linear(48, 1),
                        torch.nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.permute(0,2,1)
        x = self.pool(x).permute(0,2,1).squeeze()
        x = self.dense(x)
        return x

'''