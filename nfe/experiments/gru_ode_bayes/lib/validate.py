import torch

from nfe.experiments.gru_ode_bayes.lib.data_utils import *

def validate(model, dl, device, delta_t):
    with torch.no_grad():
        loss_val = 0
        mse_val = 0
        num_obs = 0
        for i, b in enumerate(dl):
            assert b['X_val'] is not None

            hT, _, _, _, _, p_vec = model(b['times'], b['num_obs'], b['X'].to(device), b['M'].to(device),
                                          delta_t=delta_t, cov=b['cov'].to(device), return_path=True,
                                          val_times=b['times_val'])
            m, v = torch.chunk(p_vec, 2, dim=1)

            z_reord, mask_reord = [], []
            val_numobs = torch.Tensor([len(x) for x in b['times_val']])
            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                zero_tens = torch.Tensor([0])
                z_reord.append(b['X_val'][(torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))
                                               [:-1][idx] + ind).long()])
                mask_reord.append(b['M_val'][(torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))
                                                     [:-1][idx] + ind).long()])

            X_val = torch.cat(z_reord).to(device)
            M_val = torch.cat(mask_reord).to(device)

            last_loss = (log_lik_gaussian(X_val, m, v) * M_val).sum()
            mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()

            loss_val += last_loss.cpu().numpy()
            num_obs += M_val.sum().cpu().numpy()
            mse_val += mse_loss.cpu().numpy()

        loss_val /= num_obs
        mse_val /= num_obs

    return loss_val, mse_val
