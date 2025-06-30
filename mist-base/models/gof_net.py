import torch
from typing import Optional 
from .resnet_1d import ResidualNet
from .resnet_2d import ResidualNet2D
from .unet_1d import UNet1d 
from .unet_2d import UNet


class Network1D(torch.nn.Module):
    def __init__(
        self, hidden_features=128, num_blocks=2,
        bkg=False, correlated=False, correlations_scales: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.bkg = bkg
        self.correlated = correlated
        self.correlations_scales = correlations_scales
        
        if correlated:
            self.kernel_sizes = correlations_scales + (correlations_scales % 2 == 0).int()  
            self.paddings = (self.kernel_sizes  - 1)// 2 
            self.net_list = torch.nn.ModuleList(
                 [ResidualNet(
                    in_features=1, out_features=1, 
                    hidden_features=hidden_features, num_blocks=num_blocks,
                    kernel_size=self.kernel_sizes[i].item(), padding=self.paddings[i].item()
                    ) for i in range(len(correlations_scales))]
            )
        else:
            self.net = ResidualNet(1, 1, hidden_features=hidden_features, num_blocks=num_blocks)    
            
        if bkg:
            # c = 1 if not self.correlated else len(correlations_scales)
            self.unet = UNet1d(1, 1, sizes=(4, 8, 16, 32, 64)) 
             
    def forward(self, x):
        data = x['x']
        if not self.correlated:
            if self.bkg:
                data = data - data.mean(1, keepdim=True)
                x = data - self.unet(data.unsqueeze(1)).squeeze(1)
            x = self.net(data.unsqueeze(1)).squeeze(1)
        elif self.correlated:
            if self.bkg:
                data = data - data.mean(-1, keepdim=True)
                x = data - self.unet(data)
            x = [
                self.net_list[i](data[:, i, :].unsqueeze(1)).squeeze(1) for i in range(len(self.correlations_scales))
                ]
            x = torch.stack(x, 1)
        return x
    
    
# class Network2D(torch.nn.Module):
#     def __init__(self, hidden_features=64, num_blocks=2,
#         bkg=False, correlated=False, correlations_scales: Optional[torch.Tensor] = None):
#         super().__init__()
        
#         self.bkg = bkg
#         self.correlated = correlated
#         self.correlations_scales = correlations_scales
        
#         if correlated:
#             self.kernel_sizes = correlations_scales + (correlations_scales % 2 == 0).int()  
#             self.paddings = (self.kernel_sizes  - 1)// 2 
#             self.net_list = torch.nn.ModuleList(
#                  [ResidualNet2D(
#                     in_features=1, out_features=1, 
#                     hidden_features=hidden_features, num_blocks=num_blocks,
#                     kernel_size=self.kernel_sizes[i].item(), padding=self.paddings[i].item()
#                     ) for i in range(len(correlations_scales))]
#             )
#         else:
#             self.net = ResidualNet2D(1, 1, hidden_features=hidden_features, num_blocks=num_blocks)    
            
#         if bkg:
#             c = 1 if not self.correlated else len(correlations_scales)
#             self.unet = UNet(c, c, sizes=(16, 32, 64, 128)) 
                     
#     def forward(self, x):
#         data = x['x']
#         if not self.correlated:
#             if self.bkg:
#                 data = data - self.unet(data.unsqueeze(1)).squeeze(1)       
#             x = self.net(data.unsqueeze(1)).squeeze(1)
#         elif self.correlated:
#             if self.bkg:
#                 data = data - self.unet(data)
#             x = [
#                 self.net_list[i](data[:, i, :].unsqueeze(1)).squeeze(1) for i in range(len(self.correlations_scales))
#                 ]
#             x = torch.stack(x, 1)
#         return x