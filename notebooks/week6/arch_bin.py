class Network_Epsilon_MU_det(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        ###SYMMETRY###
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        # self.logvariance_top = torch.nn.Parameter(torch.ones(Nbins)*5)
        # self.logvariance_bottom = torch.nn.Parameter(torch.ones(Nbins)*5)
        #############

        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        # self.mu_predictor = torch.nn.Sequential(
        #     torch.nn.Linear(Nbins, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 3) # 3 params: amplitude, mean, std
        # )

    def _autogauss(self, x: torch.Tensor, t) -> torch.Tensor:
        global grid
        grid = (torch.arange(x.shape[-1]).cuda())*(torch.ones_like(x).cuda()).squeeze(1)
        return t[:,0].unsqueeze(1) * torch.exp(-0.5 * ((grid.squeeze(1) - t[:,1].unsqueeze(1)) / torch.exp(t[:,2].unsqueeze(1))) ** 2)
    
    def epsilon(self,x):
        # theta_pred = self.mu_predictor(x.unsqueeze(1)).squeeze(1)
        fd = x.shape[0]
        theta_true = (torch.ones([fd,3])*torch.tensor([3,50,np.log(20)])).cuda()
        global mu_pred
        mu_pred = self._autogauss(x.unsqueeze(1),theta_true)
        global x_red
        global bank
        bank = x
        x_red = x - mu_pred
        return self.net(x_red.unsqueeze(1)).squeeze(1) + mu_pred
    
    def snr(self, x):
        return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
        ###########################################
        epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, 
                                                           device= x['x'].device, 
                                                           dtype= x['x'].dtype) - self.bounds()) * ni
        ###########################################
        
        data =  x['x0'] + epsilon_sim * ni
        
        # data = x['x']
        epsilon = self.epsilon(data)
        mask = ( x['ni'] != 0 )  
        squared_error = (epsilon - epsilon_sim)**2                                                  # [B, N_bins]
        l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
        return (l * mask.float()).sum() * 0.5
    


######## MODEL WITH WORKING SNR CORRECTION ##########
from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet

class Network_epsilon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        # self.mu_predictor = torch.nn.Sequential(
        #     torch.nn.Linear(Nbins, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 3) # 3 params: amplitude, mean, std
        # )

    def _gauss(self, x: torch.Tensor, t) -> torch.Tensor:
        ### x comes in shape [nsims, nbins]
        ### t comes in shape [nsims, 3]
        
        m = t[:,0].unsqueeze(-1)
        amp = t[:,1].unsqueeze(-1)
        sigma = t[:,2].unsqueeze(-1)

        g_grid = torch.arange(Nbins).cuda()*torch.ones_like(x).cuda()
        return amp * torch.exp(-0.5 * ((g_grid - m) / sigma) ** 2)
                
    # def epsilon(self, x):
    #     global bank
    #     bank = x
    #     x = self.net(x.unsqueeze(1)).squeeze(1) # x-net
    #     return x

    def mu(self, x):
        s_nsims, _ = x.shape
        return self._gauss(x, simulator.get_theta(s_nsims).cuda())

    def epsilon(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1) # x-net
    
    def snr(self, x):
        return self.epsilon(x-self.mu(x)) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
        ###########################################
        epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, 
                                                           device= x['x'].device, 
                                                           dtype= x['x'].dtype) - self.bounds()) * ni
        ###########################################
        
        data =  x['x0'] + epsilon_sim * ni
        data_red = data - self.mu(data)

        epsilon = self.epsilon(data_red)

        mask = ( x['ni'] != 0 )  
        squared_error = (epsilon - epsilon_sim)**2                                                  # [B, N_bins]
        l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
        return (l * mask.float()).sum() * 0.5