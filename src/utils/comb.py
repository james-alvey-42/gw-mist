import numpy as np
from scipy.ndimage import uniform_filter1d

class Comb3:
    def __init__(self, waveform, timespace, ampscale=10**2, inject=True):
        assert len(waveform) == len(timespace), 'Waveform and Waveform Time Domain must be same length!!'
        
        self.asc = ampscale
        self.length = len(waveform)

        self.w_TD_data = waveform
        self.w_TD_times = timespace

        if inject:
            self.inject_comb_stoch(waveform,timespace)
        else:
            self.load_FD(waveform,timespace)

    def load_FD(self, wf,ts):
        self.w_FD_data = np.fft.rfft(wf)
        self.w_FD_freqs = np.fft.rfftfreq(len(wf), d=np.mean(np.diff(ts)))
        self.w_FD_df = np.mean(np.diff(ts))
    
    def _inject_comb(self,wf,ts, f0,df,nf):
        global base
        base = np.zeros(len(self.w_FD_data))
        comb_freqs = np.arange(nf)*df+f0
        global indices
        indices = np.argmin(np.abs(self.w_FD_freqs[:, np.newaxis] - comb_freqs[np.newaxis, :]), axis=0)
        
        smooth = uniform_filter1d(np.abs(np.real(self.w_FD_data)), size=10)

        # f_values = np.abs(np.real(self.w_FD_data[indices]))
        f_values = smooth[indices]

        base[indices] =  np.random.normal(loc=f_values*self.asc, scale=1, size=np.shape(f_values))

        self.i_FD_data = base + self.w_FD_data
        self.epsilon = base!=0

        self.i_TD_data = np.fft.irfft(self.i_FD_data)
        self.i_TD_data_comb = np.fft.irfft(base)
    
    def inject_comb_known(self,wf,ts, f0,df,nf):
        self.load_FD(wf,ts)
        self._inject_comb(self,wf,ts, f0,df,nf)
        

    def inject_comb_stoch(self,wf,ts):
        self.load_FD(wf,ts)

        res = 10*self.w_FD_df
        nf = np.random.randint(2,10)
        f0 = np.random.uniform(low=10*res,high=(np.max(np.real(self.w_FD_freqs))-10*res))
        df = np.random.uniform(low=res, 
                               high=((np.max(np.real(self.w_FD_freqs))-f0)/nf))
        
        print(f'min {f0}, max {np.max(np.real(self.w_FD_freqs))}')
        # print(f'high {((np.max(np.real(self.w_FD_freqs))-f0)/nf)}')
        
        print(f'Generating stochastic comb res {res}, nf {nf}, f0 {f0}, df {df}')
        self._inject_comb(wf,ts, f0,df,nf)