""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.integrate import quad

########################################## Helper functions ##########################################
def tol_inverse(A):
    A = A + np.eye(A.shape[0])*(1e-6)
    return np.linalg.inv(A)# added small tolerance to avoid non-invertible matrices

def cholesky_det(A):
    A = A + np.eye(A.shape[0])*(1e-6)
    lower = np.linalg.cholesky(A)
    det_L = np.linalg.det(lower)
    return det_L**2

def single_gauss(x_s, mu, sigma):
    #take the pseudo-inverse only in case the matrix is not invertible
    exponent = -1/2*((x_s-mu).T @ tol_inverse(sigma) @ (x_s-mu))
    #The factor in front of determinant (2*np.pi)**(x_s.shape[0]/2) leads to overflow error
    #for the responsabilities it cancels out anyway
    factor = 1/(cholesky_det(sigma)**(1/2))
    return factor * np.exp(exponent)

def cost_function(x, mu_2,sigma_2,pi_2, mu_1,sigma_1,pi_1):
    log_p_2 = np.sum([np.log(np.sum([pi_2[k]*single_gauss(x[s,:],mu_2[k],sigma_2[k]) for k in range(len(pi_2))])) for s in range(x.shape[0])])
    log_p_1 = np.sum([np.log(np.sum([pi_1[k]*single_gauss(x[s,:],mu_1[k],sigma_1[k]) for k in range(len(pi_1))])) for s in range(x.shape[0])])
    cost = np.abs(log_p_1-log_p_2)
    
    return cost


def plot_GMM(mu, sigma, pi,ax):
    K = len(mu)

    # Plot each GMM component
    for k, (mean, cov, weight) in enumerate(zip(mu, sigma, pi)):
        # Plot the mean
        # Plot the covariance matrix
        M = int(np.sqrt(len(mean)))
        mean_reshaped = mean.reshape(M,M)
        ax[0,k].imshow(mean_reshaped, cmap='gray', interpolation='nearest')
        # Add a colorbar for covariance matrix
        #plt.gcf().colorbar(im, ax=ax[0,k])
        ax[0,k].set_title(f'Mean')

        # Plot the covariance matrix
        ax[1,k].imshow(cov, cmap='viridis', interpolation='nearest')
        # Add a colorbar for covariance matrix
        #plt.gcf().colorbar(im, ax=ax[1,k])

        # Set title with weight of the component
        ax[1,k].set_title(f'Covariance matrix')
    
    plt.tight_layout()

    return ax 

def plot_samples(mu, sigma, pi, ax, num):
    
    
    #TODO: drawing image, that is assigned to this component?
    for i in range(num//2):
        for j in range(2):
            #choosing random component
            prop = np.random.uniform()
            threshes = np.cumsum(pi)
            k= 0
            for u,t in enumerate(threshes):
                if prop<=t:
                    k=u
                    break

            #########
            # transforming random sample in a sample from our distribution
            sample_norm = np.random.randn(mu.shape[1])
            
            # Cholesky decomposition of the covariance matrix
            lower = np.linalg.cholesky(sigma[k,:,:]+ np.eye(sigma.shape[1])*(1e-6))
            
            # Transform standard normal variables to sample from the multivariate normal distribution
            sample = mu[k,:] + np.dot(lower, sample_norm)
            #########
            M = int(np.sqrt(len(sample)))
            mean_reshaped = sample.reshape(M,M)
            im = ax[j,i].imshow(mean_reshaped, cmap='gray', interpolation='nearest')
            # Add a colorbar for covariance matrix
            #plt.gcf().colorbar(im, ax=ax[0,k])
            ax[j,i].set_title(f'k = {k}')

    plt.tight_layout()

    return ax
########################################## Helper functions ##########################################

def task1():
    # probability density functions with change of variables, check that you obtain a valid transformed pdf
    
    """ Start of your code
    """

    
    """ End of your code
    """

def task2(x, K):
    """ Multivariate GMM

        Requirements for the plots: 
        fig1
            - ax[0,k] plot the mean of each k GMM component, the subtitle contains the weight of each GMM component
            - ax[1,k] plot the covariance of each k GMM component
        fig2 
            - ax[k,0] plot the selected *first* reshaped line of the k-th covariance matrix
            - ax[k,1] plot the selected *second* reshaped line of the k-th covariance matrix
            - ax[k,2] plot the selected *third* reshaped line of the k-th covariance matrix
            - ax[k,3] plot the selected *fourth* reshaped line of the k-th covariance matrix
        fig3: 
            - plot the 8 samples that were sampled from the fitted GMM
    """
    
    mu, sigma, pi = [], [], np.zeros((K)) # modify this later
    num_samples = 10

    fig1, ax1 = plt.subplots(2, K, figsize=(2*K,4))
    fig1.suptitle('Task 2 - GMM components', fontsize=16)

    fig2, ax2 = plt.subplots(2, num_samples//2, figsize=(2*num_samples//2,4))
    fig2.suptitle('Task 2 - samples', fontsize=16)

    """ Start of your code
    """
    S = x.shape[0]
    M = x.shape[1]
    D = M*M

    #1.) Flattening data to get proper dimensions
    x = np.reshape(x,(S, M*M))

    #2.) Initialization of covariance matrix and weights for every component
    print("Initialization of sigma and pi ...")
    sigma = [np.eye(D) for k in range(K)]
    pi = np.ones(K)/K
    print("Done \n")

    #3.) Initialization of means with k-means algorithm
    print("Initialization of mu with k-means algorithm ...")
    epsilon = 0.5
    J = 20
    j = 1
    distances = np.ones(S)*np.inf
    #pick 3 random samples as starting points for the centroids
    center_assignments = np.zeros(S)
    center_assignments_new = np.zeros(S)
    centers = [x[np.random.randint(0,S),:] for k in range(K)]

    #stopping criteria 1: max. iterations reached
    while j<=J:
        print(f"Iteration {j}")
        for s in range(S): # iterate over images
            x_s = x[s,:]
            
            for k in range(K): # iterate over clusters and find closest center
                
                if np.linalg.norm(x_s-centers[k])<distances[s]:
                    distances[s] = np.linalg.norm(x_s-centers[k])
                    center_assignments_new[s] = k

        #stopping criteria 2: centers didn't change
        if np.any(center_assignments-center_assignments_new):
            center_assignments = np.copy(center_assignments_new)
            j = j+1
        else:
            break

        #calculating new centroids
        for k in range(K):
            assigned_samples = np.where(center_assignments == k)
            new_center = np.mean(x[assigned_samples[0],:], axis=0)
            centers[k] = new_center

    mu = centers
    print("Done \n")

    #4.) EM algorithm
    print("Expectaction Maximization algorithm ...")
    e_1 = 0.5
    mu_new = np.copy(mu)
    sigma_new = np.copy(sigma)
    pi_new = np.copy(pi)


    J = 1
    j = 1

    #first iteration is always executed to get the first updated values for the cost function
    while j<= J:
        print(f"Iteration {j}")

        #applying the log sum exp trick to get responsabilities
        #denum = np.sum([pi[k]*single_gauss(x[s,:],mu[k],sigma[k]) for k in range(K)])
        
        inv_sig = [tol_inverse(sigma[k]) for k in range(K)]
        det_sig = [cholesky_det(sigma[k]) for k in range(K)]

        for k in range(K):
            w_ks_list = []

            for s in range(S):
                exponents = [-1/2*((x[s,:]-mu[k]).T @ (inv_sig[k] @ (x[s,:]-mu[k]))) for k in range(K)]
                
                y_max = np.max(exponents)
                # -> log sum trick: adding the first maximum in front
                log_w_ks_enum = np.log(pi[k])-1/2*np.log(det_sig[k]) +exponents[k]
                log_w_ks_denum = y_max + np.log(np.sum([pi[k]/np.sqrt(det_sig[k])*np.exp(exponents[k]-y_max) for k in range(K)]))
               
                w_ks = np.exp(log_w_ks_enum-log_w_ks_denum)
                w_ks_list.append(w_ks)

            w_ks_list = np.atleast_1d(np.array(w_ks_list))
            print(w_ks_list.shape)
            print(f"w_s_{k} finished with dim: {w_ks_list.shape}")

            N_k = np.sum(w_ks_list)
            print(f"N_{k} finished")
            mu_new[k] = (w_ks_list @ x)/N_k
            print(f"mu_{k} finished with dim: {mu_new[k].shape}")
            ######
            sigma_new[k] = np.einsum('ij,ik,i->jk', x - mu[k], x - mu[k], w_ks_list) / N_k
            print(f"sigma_{k} finished")
            pi_new[k] = N_k/S
            print(f"pi_{k} finished")


        #cost = cost_function(x,mu_new, sigma_new, pi_new,mu,sigma, pi)
        cost = 1


        if (cost >= e_1):
            print(f"Remaining cost: {cost} higher than {epsilon}")
            mu = np.copy(mu_new)
            sigma = np.copy(sigma_new)
            pi = np.copy(pi_new)

            j = j+1
            
        else:
            print("Algorithm converged.")
            mu = np.copy(mu_new)
            sigma = np.copy(sigma_new)
            pi = np.copy(pi_new)
            break
        print(f"Finished iteration {j}")

        #Plotting the GMM components
        ax1 = plot_GMM(mu_new, sigma_new, pi_new, ax1)

        fig1.show()

        #drawing random samples
        ax2 = plot_samples(mu_new, sigma_new, pi_new, ax2,num_samples)
        
        fig2.show()

    """ End of your code
    """

    for k in range(K):
        ax1[0,k].set_title('C%i with %.2f' %(k,pi[k])), ax1[0,k].axis('off'), ax1[1,k].axis('off')

    return (mu, sigma, pi), (fig1,fig2)

def task3(x, mask, m_params):
    """ Conditional GMM

        Requirements for the plots: 
        fig
            - ax[s,0] plot the corrupted test sample s
            - ax[s,1] plot the restored test sample s (by using the posterior expectation)
            - ax[s,2] plot the groundtruth test sample s 
    """
    
    S, sz, _ = x.shape

    fig, ax = plt.subplots(S,3,figsize=(3,8))
    fig.suptitle('Task 3 - Conditional GMM', fontsize=12)
    for a in ax.reshape(-1):
        a.axis('off')
        
    ax[0,0].set_title('Condition',fontsize=8), ax[0,1].set_title('Posterior Exp.',fontsize=8), ax[0,2].set_title('Groundtruth',fontsize=8)
    for s in range(S):
        ax[s,2].imshow(x[s], vmin=0, vmax=1., cmap='gray')

    """ Start of your code
    """



    """ End of your code
    """

    return fig

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # Task 1: transformations of pdfs
    task1()

    # load train and test data
    with np.load("data.npz") as f:
        x_train = f["train_data"]
        x_test = f["test_data"]

    # Task 2: fit GMM to FashionMNIST subset
    K = 3 # TODO: adapt the number of GMM components
    gmm_params, fig1 = task2(x_train,K)

    # Task 3: inpainting with conditional GMM
    mask = None
    fig2 = task3(x_test,mask,gmm_params)

    for f in fig1:
        pdf.savefig(f)
    pdf.savefig(fig2)
    pdf.close()
    