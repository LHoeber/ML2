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
    num_iterations =200
    learning_rate = 0.01
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
        x_sigma.requires_grad_(True)
        input_samples = torch.cat((x_sigma, sigma_vals), dim=1)
        
        #input_samples.requires_grad_(True)

        '''sig_tensor = torch.full((x_tensor.size(0), 1), noiselevel, dtype=torch.float32)
            input = torch.cat((x_tensor, y_tensor, sig_tensor), dim=1)
            input.requires_grad_(True)

            # bw pass
            output = Simple_NN(input).reshape(res, res)
            energy[:, :] = output.detach().numpy()
            result = Simple_NN(input)
            result.backward(torch.ones_like(result))

            # scores
            x_scores = input.grad[:, 0].reshape(res, res)
            y_scores = input.grad[:, 1].reshape(res, res)
            scores[:, :, 0] = x_scores.detach().numpy()
            scores[:, :, 1] = y_scores.detach().numpy()'''

        # forwards pass
        output_NN = Simple_NN.forward(input_samples)
        #TODO: am i supposed to use autograd here

        
        first_term = -(x_sigma - x)/sigma_vals**2
        grad_output_NN = torch.ones_like(output_NN)
        second_term = -torch.autograd.grad(output_NN,x_sigma,grad_outputs=grad_output_NN,create_graph=True)[0]
        loss = torch.linalg.norm(first_term-second_term[:,:2])
        
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
    #plt.show()


    #4. computing the score and plotting it

    #helper to convert to numpy array
    def torch2num(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        else:
            return x
       

    # function for analytically derived score
    def GMM_score(mu, sig, x):
        
        mu = torch2num(mu)
        sig = torch2num(sig)
        x = torch2num(x)

        sigma_inv = np.linalg.inv(sig)

        

        return - np.dot(sigma_inv,(x - mu))
    
    # function for probability with multivariate gaussian
    def multivar_Gauss_pdf(mu, sig, x):
        D = len(mu)
        sig_det = np.linalg.det(sig)
        factor = np.power((2 * np.pi), float(D) / 2) * np.sqrt(sig_det)
        sig_inv = np.linalg.inv(sig)

        # check types
        if isinstance(mu, torch.Tensor):
            mu = mu.numpy()
        if isinstance(sig, torch.Tensor):
            sig = sig.numpy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        exponent = -0.5 * np.dot((x - mu).T, np.dot(sig_inv, x - mu))
        return 1/factor * np.exp(exponent)

    # plot (same style as example)
   

    def GMM_scores_plot_and_calculate(x, mus, sigmas, chosen_noise_levels, res=32):
        # center grid
        xlims = (min(x[:, 0]), max(x[:, 0]))
        ylims = (min(x[:, 1]), max(x[:, 1]))
        X, Y = np.meshgrid(np.linspace(xlims[0],xlims[1], res), np.linspace(ylims[0],ylims[1], res))
        data = np.stack([X, Y], axis=-1)
        
        scores_with_noise = []

        #calculate all components of GMM for every noise level
        for l, noise_l in enumerate(chosen_noise_levels):
            noise_l = torch2num(noise_l)
            pdf_vals = np.zeros((res, res))
            scores_all_components = np.zeros((res, res, 2))

            #iteration over gauss components
            for mu, sig in zip(mus, sigmas):
                sig=torch2num(sig)
                mu =torch2num(mu)
                sig_noise = sig + np.eye(2) * noise_l**2
                for m in range(res):
                    for n in range(res):
                        point = data[m,n,:]
                        #probability of the point, given our GMM(for one component)
                        amplitude = multivar_Gauss_pdf(mu, sig_noise, point)
                        pdf_vals[m, n] += amplitude
                        #score(normalized to 1, so it only shows direction?) weighted with the pdf
                        scores_all_components[m, n,:] += GMM_score(mu, sig_noise, point) * amplitude
            
            scores_with_noise.append(scores_all_components)

            #plotting
            ax4[0, l].contourf(X, Y, pdf_vals, levels=100)
            color_map = np.linalg.norm(scores_all_components,axis = 2)
            ax4[1, l].quiver(X, Y, scores_all_components[:, :, 0], scores_all_components[:, :, 1], color_map)
            

        
        '''
                sig_k = sig_k + torch.eye(2) * noise_l**2
                normal_part = Gaussian(data,mu_k,sig_k)
                inner_deriv = ((mu_k-data)@np.linalg.inv(sig_k))
                new_component = np.multiply(normal_part[:, np.newaxis], inner_deriv.numpy())

                new_component = new_component.reshape((res,res))
                denominator = denominator + new_component
            
                enumerator = np.random.normal(x,mu_k.numpy(),sig_k.numpy())*((mu_k-x)@np.linalg.inv(sig_k))
               
            score_all_components = enumerator/denominator
            scores_with_noise.append(score_all_components)
 
        #getting colors of the arrows:
        #TODO: take GMM results for this(with noise(same as already donee for 2))
        for l,(score,noise_level) in enumerate(zip(scores_with_noise,sigma_list)):
            x_sigma = x + noise_level * torch.randn(x.shape)

            # plotting
            ax4[0, i].contourf(X, Y, x_sigma, levels=100)
            ax4[1, i].quiver(X, Y, score[l][:, :, 0], score[l][:, :, 1])#, np.hypot(scores[:, :, 0], scores[:, :, 1]))
        '''
        '''
        for i, noiselevel in enumerate(chosen_sigmas):
            #density = amplitude/color of the arrows
            density = np.zeros((res, res))
            scores = np.zeros((res, res, 2))
            
            for m, sig in zip(mu, sigma):
                disturbed_sig = sig + torch.eye(2) * noiselevel**2
                for j in range(res):
                    for k in range(res):
                    #     p = np.array([X[j, k], Y[j, k]])
                    #     density_p = GMM(m, disturbed_sig, p)
                    #     density[j, k] += density_p
                        scores[j, k] += GMM_scores(m, disturbed_sig, p) * density_p

            # plot 
            ax_density = ax4[0, i]
            ax_density.contourf(X, Y, density, levels=100)
            ax_scores = ax4[1, i]
            ax_scores.quiver(X, Y, scores[:, :, 0], scores[:, :, 1], np.hypot(scores[:, :, 0], scores[:, :, 1]))
    
        '''
    chosen_sigmas = [sigma_list[0],sigma_list[int(len(sigma_list)/2)], sigma_list[-1]]
    GMM_scores_plot_and_calculate(x, mu, sig, chosen_sigmas, 32)


    #5. Compare energy based model with analytical counterpart (density and scores)
    def compare_energy_score(Simple_NN, x, chosen_sigmas, res = 32):
        xlims = (min(x[:, 0]), max(x[:, 0]))
        ylims = (min(x[:, 1]), max(x[:, 1]))
        X, Y = np.meshgrid(np.linspace(xlims[0],xlims[1], res), np.linspace(ylims[0],ylims[1], res))
        data = np.stack([X, Y], axis=-1)
        x_tensor = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)

        # save energy and scores
        energy = np.zeros((res, res))
        scores = np.zeros((res, res, 2))

        for l, noiselevel in enumerate(chosen_sigmas):
            # create noise tensor
            sig_tensor = torch.full((x_tensor.size(0), 1), noiselevel, dtype=torch.float32)
            input = torch.cat((x_tensor, y_tensor, sig_tensor), dim=1)
            input.requires_grad_(True)

            # bw pass
            output = Simple_NN(input).reshape(res, res)
            energy[:, :] = output.detach().numpy()
            result = Simple_NN(input)
            result.backward(torch.ones_like(result))

            # scores
            x_scores = input.grad[:, 0].reshape(res, res)
            y_scores = input.grad[:, 1].reshape(res, res)
            scores[:, :, 0] = x_scores.detach().numpy()
            scores[:, :, 1] = y_scores.detach().numpy()

            # plotting
            ax5[0, l].contourf(X, Y, energy, levels=100)
            color_map = np.linalg.norm(scores,axis = 2)
            ax5[1, l].quiver(X, Y, scores[:, :, 0], scores[:, :, 1], color_map)
        

    compare_energy_score(Simple_NN, x, chosen_sigmas,32)

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
    T =  100     #number of Langevin samples per noise level
    S = n_samples

    #generating the initial samples:
    #TODO: from which distribution should the samples be taken?
    #the perturbed GMM, or simple standard normal dist?
    x0 = torch.randn(5000, 2)

    
    # sigma_indices = np.random.randint(0, L, x.shape[0])
    # sigma_vals = torch.tensor([sigma_list[idx] for idx in sigma_indices])
    # sigma_vals = sigma_vals.reshape((x.shape[0],1)).float()


    for i in range(len(sigma_list)-1,-1,-1):
        sigma_i = sigma_list[i].float()
        sigma_vals = torch.ones((x0.shape[0],1))*(sigma_i).float()#.view(-1, 1).clone().detach()
        print(x0.shape[0])
        #sigma_vals = sigma_vals.reshape((x0.shape[0],1)).float()

        x = x0.clone()
        x.requires_grad_(True)
        x = torch.cat((x, sigma_vals), dim=1)
        

        sigma = sigma_list[i].float()
        alpha = eps*sigma**2/sigma_list[0].float()**2
        for t in range(T):
            z = torch.randn_like(x)

            # x_scores = input.grad[:, 0].reshape(res, res)
            # y_scores = input.grad[:, 1].reshape(res, res)
            # scores[:, :, 0] = x_scores.detach().numpy()
            # scores[:, :, 1] = y_scores.detach().numpy()

            output_NN = Simple_NN(x).detach()
            grad_output_NN = torch.ones_like(output_NN)
            score = torch.autograd.grad(output_NN,x,grad_outputs=grad_output_NN,create_graph=True)[0]
            #TODO: use the score of the simpleNN istead of the simple_NN itself
            x = x - (alpha / 2) * score + torch.sqrt(alpha) * z

    #plotting
    print("langevin complete")
    bins = 128
    ax6[0].hist2d(x0[:,0].detach().cpu().numpy(), x0[:,1].detach().cpu().numpy(), bins=bins, cmap='viridis')
    ax6[1].hist2d(x[:,0].detach().cpu().numpy(), x[:,1].detach().cpu().numpy(), bins=bins, cmap='viridis')
    ax6[0].set_title('Initial samples (standard normal)')
    ax6[1].set_title('Generated samples using Langevin dynamics')
    

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
    #fig6 = sampling(Simple_NN=Simple_NN, sigma_list=sigma_list, n_samples=5000)

    pdf.savefig(fig1)
    for f in figs:
        pdf.savefig(f)
    #pdf.savefig(fig6)

    pdf.close()
    
