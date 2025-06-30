import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gw150814_simulator import GW150814, defaults
import torch
from tqdm import tqdm


simulator = GW150814(defaults)
# Downsample factor 8, mask between -0.1-0.1 seconds --> 102 bins


# N simulations = 1010000
N = simulator.posterior_array.shape[0]
posterior_samples = torch.zeros(N, 102)
for i in tqdm(range(N)):
    params = simulator.posterior_array[i]
    theta_ripple = jnp.array(
        [
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
        ]
    )  # M_c, eta = q / (1 + q)^2, s1_z, s2_z, d_L, t_c + epoch, phase_c, iota
    ra, dec, psi = params[8], params[9], params[10]
    hp, hc = simulator.call_waveform(theta_ripple)
    hdet_fd = simulator.detector.fd_response(
        simulator.frequencies,
        {"p": hp, "c": hc},
        params={"ra": ra, "dec": dec, "psi": psi, "gmst": simulator.gmst},
    )
    # strain = simulator.frequency_to_time_domain(hdet_fd * simulator.filter)
    strain = simulator.whitened_frequency_to_time_domain(hdet_fd * simulator.filter)
    posterior_samples[i] = simulator._jax_to_torch(simulator._process(strain))
torch.save(posterior_samples, "stores/gw150814_white_20-1024Hz_post_d8_m1_1010k.pt")


NN = 10_000_000
noise_samples = torch.zeros(NN, 102)
for i in tqdm(range(NN)):
    noise = simulator.generate_time_domain_noise()
    noise_samples[i] = simulator._jax_to_torch(simulator._process(noise))
torch.save(noise_samples, "stores/gw150814_white_20-1024Hz_noise_d8_m1_10M.pt")