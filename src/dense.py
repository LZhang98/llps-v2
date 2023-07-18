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
    
    def extract_features(self, x):
        x = x.permute(0,2,1)
        a = self.pool(x).permute(0,2,1).squeeze() # pool
        b = self.dense[0](a) # Lin 1
        c = self.dense[1](b) # ReLU
        c = self.dense[2](c) # Lin 2
        d = self.dense[3](c)

        # [pool output, lin 1 output, lin 2 output, final output]
        return [a, b, c, d]


# class HybridClassifier (torch.nn.Module):
#     def __init__(self, total_dim) -> None:
#         super().__init__()
#         self.pool = torch.nn.AdaptiveMaxPool1d(1)
#         self.dense = torch.nn.Sequential(
#             torch.nn.Linear(total_dim, 48)
#         )
    
#     def foward(self, x):
#         return x