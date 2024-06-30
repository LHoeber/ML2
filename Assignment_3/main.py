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
    SHOW_PLOTS = True
    def Gauss(xs,mu,sig):
        factor = 1/(2*np.pi*np.sqrt(np.linalg.det(sig)))
        exponent = -1/2*((xs-mu).T @ np.linalg.inv(sig) @(xs-mu))
        return factor * np.exp(exponent)
    
    def GMM(xs,pis,mus, sigs):
        # xs... one sample
        K = len(pis)
        S = x.shape[0]
        prob = np.sum([Gauss(xs,mus[k,:],sigs[k,:,:]) for k in range(0,K)])
        return prob

    K = 3
    S = 10000
    mu_1 = 1/4*torch.tensor([1,1])
    mu_2 = 1/4*torch.tensor([3,1])
    mu_3 = 1/4*torch.tensor([2,3])
    mu = torch.stack([mu_1,mu_2,mu_3])

    sig_1 = 0.01*torch.eye(2)
    sig_2 = torch.clone(sig_1)
    sig_3 = torch.clone(sig_1)
    sig = torch.stack([sig_1,sig_2,sig_3])

    a = 1/3*torch.ones(K)

    #1.) generating data for interval [0,4] in both dimensions
    '''x_lims = [0,1]
    y_lims = [0,1]

    #uniformly drawing sample points in both directions and
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(S, 2))
    rand_probs = [GMM(x[s,:],a,mu,sig) for s in range(0,S)]

    bins = 128
    # Plot the 2D histogram
    plt.hist2d(x[:,0], x[:,1], bins=bins, weights=rand_probs, cmap='inferno')
    plt.colorbar(label='Probability Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Histogram with Probabilities')
    plt.show()'''

    def sample_GMM(mu, sigma):
        L = torch.cholesky(sigma)  # for more efficiency bzw. because we need sqrt
        Z = torch.randn(mu.shape[0]) # standard normal distributed data, that will get transformed
        return mu + L@Z

    # Generating samples
    x = torch.zeros((n_samples, 2))

    for i in range(n_samples):
        # choose random component
        k = np.random.choice(K, p=np.array(a))
        
        # Sampling from component
        x[i, :] = sample_GMM(mu[k], sig[k])

    bins = 128
    # Plotting all the random samples
    plt.hist2d(x[:,0], x[:,1], bins=bins, cmap='viridis')
    plt.colorbar(label='number of samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Histogram for randomly generated samples')
    plt.axis("equal")
    plt.tight_layout()
    if SHOW_PLOTS:
        fig1.show()
    

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

    # plot analytic density/scores (fig4) vs. learned by Simple_NNwork (fig5)
    fig4, ax4 = plt.subplots(2,3,figsize=(16,10))
    fig5, ax5 = plt.subplots(2,3,figsize=(16,10))

    mu, sig, a = params

    """ Start of your code
    """
    SHOW_PLOTS = False
    #3 Denoising score matching in practice
    #1. Data distribution with different noise levels
    sigma_1 = 0.1
    sigma_L = 0.8
    L = 10
    sigma_list = get_sigmas(sigma_1, sigma_L, L)
    chosen_sigmas = [0,sigma_list[0], sigma_list[-1]]

    #Plotting original dist, dist with sigma_1, dist with sigma_L 
    bins = 128
    for i,sigma in enumerate([0,sigma_1,sigma_L]):
        x_sigma = x + sigma * torch.randn(x.shape)
        ax2[i].hist2d(x_sigma[:,0], x_sigma[:,1], bins=bins, cmap='viridis')
    if SHOW_PLOTS:
        plt.show()

    #2. creating a small and simple neural Simple_NNwork
    # (like in given example from Tutorial, but one extra layer)
    
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.W1 = nn.Linear(input_size, hidden_size, bias=True) 
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.W3 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.W4 = nn.Linear(hidden_size, output_size, bias=True)
            self.elu = nn.ELU()
        
        def forward(self, input):
            tmp = self.elu(self.W1(input))
            tmp = self.elu(self.W2(tmp))
            tmp = self.elu(self.W3(tmp))
            return self.W4(tmp)
        
    Simple_NN = SimpleMLP(3,128,1)

    # getting number of learnable params
    num_params = sum(p.numel() for p in Simple_NN.parameters() if p.requires_grad)
    print("Number of learnable parameters: ", num_params)


    #3. Training the neural network
    num_iterations =500
    learning_rate = 0.05
    adam = optim.Adam(Simple_NN.parameters(), lr=learning_rate)
    losses = []

    for i in range(num_iterations):
        # add a random noise level to each data sample, to make input 3-dimensional
        sigma_indices = np.random.randint(0, L, x.shape[0])
        sigma_vals = torch.tensor([sigma_list[idx] for idx in sigma_indices])
        sigma_vals = sigma_vals.reshape((x.shape[0],1)).float()

        #do all the sampling points in each iteration need to get same noise level
        #or does the level need to get chosen for ever point separately?
        x_sigma = x + sigma_vals * torch.randn(x.shape)
        input_samples = torch.cat((x_sigma, sigma_vals), dim=1)

        # forwards pass
        score_estimate = Simple_NN.forward(input_samples).squeeze()
        score_target = -(x_sigma - x).norm(dim=1)/sigma_vals.squeeze()**2
        #loss = F.binary_cross_entropy(torch.sigmoid(score_estimate), F.one_hot(score_target.long()).float()) # ((prediction - y_train)**2).mean()
        # calculate loss
        loss = nn.MSELoss()(score_estimate, score_target)

        # backwards pass and optimization
        adam.zero_grad()
        loss.backward()
        adam.step()

        losses.append(loss.item())
        print(f"Training iteration: {i} of {num_iterations}")

    # ploting
    ax3.plot(losses)
    ax3.set_xlabel('iterations')
    ax3.set_ylabel('loss')
    ax3.set_title(f'MSE with final loss: {losses[-1]}')
    plt.show()


    #4. computing the score and plotting it
    def score_gmm(mu, sigma, x):
        sigma_inv = np.linalg.inv(sigma.numpy())
        
        return -np.dot(sigma_inv, x - mu.numpy())

    # calculate GMM density
    def density_gmm(mu, sigma, x):
        dims = len(mu)
        sigma_det = np.linalg.det(sigma)
        norm = 1.0 / (np.power((2 * np.pi), float(dims) / 2) * np.sqrt(sigma_det))
        sigma_inv = np.linalg.inv(sigma)

        mu = mu.numpy()
        sigma = sigma.numpy()

        result = np.exp(-0.5 * np.dot((x - mu).T, np.dot(sigma_inv, x - mu)))
        return norm * result

    # plot (same style as example)
    def plot_gmm_scores(x, mu, sigma, chosen_sigmas, grid_size=32):
        # center grid
        x_min, x_max = x[:, 0].min(), x[:, 0].max()
        y_min, y_max = x[:, 1].min(), x[:, 1].max()
        x_mg, y_mg = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

        for i, noiselevel in enumerate(chosen_sigmas):
            # save density and noise per level
            density = np.zeros((grid_size, grid_size))
            scores = np.zeros((grid_size, grid_size, 2))
            
            for m, sig in zip(mu, sigma):
                sigma_n = sig + torch.eye(2) * noiselevel**2
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


    plot_gmm_scores(x, mu, sig, chosen_sigmas)
    print("3.4) DONE: plot_gmm_scores")



    # -----   Task 3.5 -----
    # energy vs scores
    def eval_energy(model, x, chosen_sigmas, grid_size=32):
        x_min, x_max = x[:, 0].min(), x[:, 0].max()
        y_min, y_max = x[:, 1].min(), x[:, 1].max()
        x_mg, y_mg = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        x_tensor = torch.tensor(x_mg.reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_mg.reshape(-1, 1), dtype=torch.float32)

        # save energy and scors
        energy = np.zeros((grid_size, grid_size))
        scores = np.zeros((grid_size, grid_size, 2))

        for n, noiselevel in enumerate(chosen_sigmas):
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

    energy, scores = eval_energy(Simple_NN, x, chosen_sigmas)
    print("3.5) DONE: eval_energy (and plotted)")
    """ End of your code
    """

    for idx, noiselevel in enumerate(chosen_sigmas):
        ax4[0,idx].set_title(r'$\sigma$=%f' %noiselevel)
        ax5[0,idx].set_title(r'$\sigma$=%f' %noiselevel)

        ax4[0,idx].set_xticks([]), ax4[0,idx].set_yticks([])
        ax4[1,idx].set_xticks([]), ax4[1,idx].set_yticks([])
        ax5[0,idx].set_xticks([]), ax5[0,idx].set_yticks([])
        ax5[1,idx].set_xticks([]), ax5[1,idx].set_yticks([])

    ax4[0,0].set_ylabel('analytic density'), ax4[1,0].set_ylabel('analytic scores')
    ax5[0,0].set_ylabel('learned density'), ax5[1,0].set_ylabel('learned scores')

    return Simple_NN, sigma_list, (fig2, fig3, fig4, fig5)

def sampling(Simple_NN, sigma_list, n_samples):
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
    eps = 0.1 #step size adjustment across noise levels
    T =  50     #number of Langevin samples per noise level
    x0 = torch.randn((n_samples, 2))   #initial sample
    x = x0.clone()

    for i in range(len(sigma_list)-1,-1,-1):
        sigma = sigma_list[i].float()
        alpha = eps*sigma**2/sigma_list[0].float()**2
        for t in range(T):
            z = torch.randn_like(x)
            x.requires_grad_(True)
            score = Simple_NN(torch.cat((x, sigma.expand(n_samples, 1)), dim=1)).detach()
            x = x - (alpha / 2) * score + torch.sqrt(alpha) * z
    #plotting
    bins = 128
    #].hist2d(x0[:,0].cpu().numpy(), x0[:,1].cpu().numpy(), bins=bins, cmap='viridis')
    #ax6[1].hist2d(x[:,0].cpu().numpy(), x[:,1].cpu().numpy(), bins=bins, cmap='viridis')
    #ax6[0].set_title('Initial samples (standard normal)')
    #ax6[1].set_title('Generated samples using Langevin dynamics')
    

    """ End of your code
    """

    return fig6 

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # generate data
    x, params, fig1 = generate_data(n_samples=10000)

    # denoising score matching
    Simple_NN, sigma_list, figs = dsm(x=x, params=params)

    # sampling
    fig6 = sampling(Simple_NN=Simple_NN, sigma_list=sigma_list, n_samples=5000)

    pdf.savefig(fig1)
    for f in figs:
        pdf.savefig(f)
    pdf.savefig(fig6)

    pdf.close()
    
