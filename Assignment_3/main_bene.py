""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib.backends.backend_pdf import PdfPages

def get_sigmas(sigma_1, sigma_L, L):
    # geometric progression for noise levels from \sigma_1 to \sigma_L 
    return torch.tensor(np.exp(np.linspace(np.log(sigma_1),np.log(sigma_L), L)))

def generate_data(n_samples):
    """ Generate data from 3-component GMM

        Requirements for the plot: 
        fig1 
            - this plot should contain a 2d histogram of the generated data samples

    """
    x, mu, sig, a = torch.zeros((n_samples,2)), None, None, None
    fig1 = plt.figure(figsize=(5,5))
    plt.title('Data samples')

    """ Start of your code
    """
    
    
    """ End of your code
    """

    return x, (mu, sig, a), fig1

def dsm(x, params):
    """ Denoising score matching
    
        Requirements for the plots:
        fig2
            - ax2[0] contains the histogram of the data samples
            - ax2[1] contains the histogram of the data samples perturbed with \sigma_1
            - ax2[2] contains the histogram of the data samples perturbed with \sigma_L
        fig3
            - this plot contains the log-loss over the training iterations
        fig4
            - ax4[0,0] contains the analytic density for the data samples perturbed with \sigma_1
            - ax4[0,1] contains the analytic density for the data samples perturbed with an intermediate \sigma_i 
            - ax4[0,2] contains the analytic density for the data samples perturbed with \sigma_L

            - ax4[1,0] contains the analytic scores for the data samples perturbed with \sigma_1
            - ax4[1,1] contains the analytic scores for the data samples perturbed with an intermediate \sigma_i 
            - ax4[1,2] contains the analytic scores for the data samples perturbed with \sigma_L
        fig5
            - ax5[0,0] contains the learned density for the data samples perturbed with \sigma_1
            - ax5[0,1] contains the learned density for the data samples perturbed with an intermediate \sigma_i 
            - ax5[0,2] contains the learned density for the data samples perturbed with \sigma_L

            - ax5[1,0] contains the learned scores for the data samples perturbed with \sigma_1
            - ax5[1,1] contains the learned scores for the data samples perturbed with an intermediate \sigma_i 
            - ax5[1,2] contains the learned scores for the data samples perturbed with \sigma_L
    """
    
    fig2, ax2 = plt.subplots(1,3,figsize=(10,3))
    ax2[0].hist2d(x.cpu().numpy()[:,0],x.cpu().numpy()[:,1],128), ax2[0].set_title(r'data $x$')
    ax2[1].set_title(r'data $x$ with $\sigma_{1}$')
    ax2[2].set_title(r'data $x$ with $\sigma_{L}$')

    fig3, ax3 = plt.subplots(1,1,figsize=(5,3))
    ax3.set_title('Log loss over training iterations')

    # plot analytic density/scores (fig4) vs. learned by network (fig5)
    fig4, ax4 = plt.subplots(2,3,figsize=(16,10))
    fig5, ax5 = plt.subplots(2,3,figsize=(16,10))

    mu, sig, a = params

    """ Start of your code
    """

    # -----  Task 3.1   -----
    # noise levels
    sigma_1 = 0.1
    sigma_L = 0.5
    L = 12
    sigmas_all = get_sigmas(sigma_1, sigma_L, L)
    noiselevels = [sigma_1, (sigma_1 + sigma_L) / 2, sigma_L]


    # plot  the perturbed data
    for sigma, ax in zip([sigma_1, sigma_L], [ax2[1], ax2[2]]):
        x_perturbed = x + sigma * torch.randn_like(x)
        ax.hist2d(x_perturbed.cpu().numpy()[:,0], x_perturbed.cpu().numpy()[:,1], 128)
    print("3.1) DONE: noise levels and plotted")


    # -----  Task 3.2   -----
    # simple NN
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 128)  
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128) 
            self.fc4 = nn.Linear(128, 1)
            self.elu = nn.ELU()
        
        # overwrite fw
        def forward(self, x):
            x = self.elu(self.fc1(x))
            x = self.elu(self.fc2(x))
            x = self.elu(self.fc3(x))
            x = self.fc4(x)
            return x


    Net = SimpleNN()

    # learnable params
    num_parameters = sum(p.numel() for p in Net.parameters() if p.requires_grad)
    print("learnable parameters: ", num_parameters)
    print("3.2) DONE: trained and loss plotted")



    # -----  Task 3.3   -----
    # init params
    num_iterations = 800
    learning_rate = 0.005
    adam_optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
    losses_save = []

    for iteration in range(num_iterations):
        adam_optimizer.zero_grad()

        # sample rand noise
        idx_sigma = np.random.randint(0, L, size=x.shape[0])
        sigma = torch.tensor(sigmas_all[idx_sigma]).float().view(-1, 1).clone().detach()
        noise = sigma * torch.randn_like(x)
        x_perturbed = x + noise
        input_data = torch.cat((x_perturbed, sigma), dim=1)

        # forwards pass
        out = Net(input_data)

        # calclate loss
        score = -out.squeeze()
        score_target = -(x_perturbed - x).norm(dim=1) / (sigma.squeeze() ** 2) 
        loss = nn.MSELoss()(score, score_target)

        # backwards pass and optimization
        loss.backward()
        adam_optimizer.step()

        losses_save.append(loss.item())
        print(f"iteration: {iteration}/{num_iterations} | loss: {loss}")

    # plot
    ax3.plot(losses_save)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Loss')
    print("3.3) DONE: trained and loss plotted")



    # -----   Task 3.4 -----
    # calculate GMM score
    def score_gmm(mu, sigma, x):
        sigma_inv = np.linalg.inv(sigma)
        
        # check types
        if isinstance(mu, torch.Tensor):
            mu = mu.numpy()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.numpy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return -np.dot(sigma_inv, x - mu)

    # calculate GMM density
    def density_gmm(mu, sigma, x):
        dims = len(mu)
        sigma_det = np.linalg.det(sigma)
        norm = 1.0 / (np.power((2 * np.pi), float(dims) / 2) * np.sqrt(sigma_det))
        sigma_inv = np.linalg.inv(sigma)

        # check types
        if isinstance(mu, torch.Tensor):
            mu = mu.numpy()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.numpy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        result = np.exp(-0.5 * np.dot((x - mu).T, np.dot(sigma_inv, x - mu)))
        return norm * result

    # plot (same style as example)
    def plot_gmm_scores(x, mu, sigma, noiselevels, grid_size=32):
        # center grid
        x_min, x_max = x[:, 0].min(), x[:, 0].max()
        y_min, y_max = x[:, 1].min(), x[:, 1].max()
        x_mg, y_mg = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

        for i, noiselevel in enumerate(noiselevels):
            # save density and noise per level
            density = np.zeros((grid_size, grid_size))
            scores = np.zeros((grid_size, grid_size, 2))
            
            for m, sig in zip(mu, sigma):
                sigma_n = sig + np.eye(2) * noiselevel**2
                for j in range(grid_size):
                    for k in range(grid_size):
                        p = np.array([x_mg[j, k], y_mg[j, k]])
                        density_p = density_gmm(m, sigma_n, p)
                        density[j, k] += density_p
                        scores[j, k] += score_gmm(m, sigma_n, p) * density_p

            # plot 
            ax_density = ax4[0, i]
            ax_density.contourf(x_mg, y_mg, density, levels=100)
            ax_scores = ax4[1, i]
            ax_scores.quiver(x_mg, y_mg, scores[:, :, 0], scores[:, :, 1], np.hypot(scores[:, :, 0], scores[:, :, 1]))


    plot_gmm_scores(x, mu, sig, noiselevels)
    print("3.4) DONE: plot_gmm_scores")



    # -----   Task 3.5 -----
    # energy vs scores
    def eval_energy(model, x, noiselevels, grid_size=32):
        x_min, x_max = x[:, 0].min(), x[:, 0].max()
        y_min, y_max = x[:, 1].min(), x[:, 1].max()
        x_mg, y_mg = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        x_tensor = torch.tensor(x_mg.reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_mg.reshape(-1, 1), dtype=torch.float32)

        # save energy and scors
        energy = np.zeros((grid_size, grid_size))
        scores = np.zeros((grid_size, grid_size, 2))

        for n, noiselevel in enumerate(noiselevels):
            # create noise tensor
            sig_tensor = torch.full((x_tensor.size(0), 1), noiselevel, dtype=torch.float32)
            input = torch.cat((x_tensor, y_tensor, sig_tensor), dim=1)
            input.requires_grad_(True)

            # bw pass
            output = model(input).reshape(grid_size, grid_size)
            energy[:, :] = output.detach().numpy()
            res = model(input)
            res.backward(torch.ones_like(res))

            # score
            x_scores = input.grad[:, 0].reshape(grid_size, grid_size)
            y_scores = input.grad[:, 1].reshape(grid_size, grid_size)
            scores[:, :, 0] = x_scores.detach().numpy()
            scores[:, :, 1] = y_scores.detach().numpy()

            # plot
            ax5_energy = ax5[0, n]
            ax5_scores = ax5[1, n]
            ax5_energy.contourf(x_mg, y_mg, energy, levels=100)
            ax5_scores.quiver(x_mg, y_mg, scores[:, :, 0], scores[:, :, 1], np.hypot(scores[:, :, 0], scores[:, :, 1]))

        return energy, scores

    energy, scores = eval_energy(Net, x, noiselevels)
    print("3.5) DONE: eval_energy (and plotted)")

    """ End of your code
    """

    for idx, noiselevel in enumerate(noiselevels):
        ax4[0,idx].set_title(r'$\sigma$=%f' %noiselevel)
        ax5[0,idx].set_title(r'$\sigma$=%f' %noiselevel)

        ax4[0,idx].set_xticks([]), ax4[0,idx].set_yticks([])
        ax4[1,idx].set_xticks([]), ax4[1,idx].set_yticks([])
        ax5[0,idx].set_xticks([]), ax5[0,idx].set_yticks([])
        ax5[1,idx].set_xticks([]), ax5[1,idx].set_yticks([])

    ax4[0,0].set_ylabel('analytic density'), ax4[1,0].set_ylabel('analytic scores')
    ax5[0,0].set_ylabel('learned density'), ax5[1,0].set_ylabel('learned scores')

    return Net, sigmas_all, (fig2, fig3, fig4, fig5)

def sampling(Net, sigmas_all, n_samples):
    """ Sampling from the learned distribution
    
        Requirements for the plots:
            fig6
                - ax6[0] contains the histogram of the data samples
                - ax6[1] contains the histogram of the generated samples
    
    """
    
    fig6, ax6 = plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    ax6[0].set_title(r'data $x$')
    ax6[1].set_title(r'samples')

    """ Start of your code
    """
   
        

    """ End of your code
    """

    return fig6 

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # generate data
    x, params, fig1 = generate_data(n_samples=10000)

    # denoising score matching
    Net, sigmas_all, figs = dsm(x=x, params=params)

    # sampling
    fig6 = sampling(Net=Net, sigmas_all=sigmas_all, n_samples=5000)

    pdf.savefig(fig1)
    for f in figs:
        pdf.savefig(f)
    pdf.savefig(fig6)

    pdf.close()
    print("DONE: figures saved")
    
