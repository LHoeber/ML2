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
    #add small offset to main diagonal
    return np.linalg.inv(A+np.eye(A.shape[1])*(1e-6))

def log_cholesky_det_sqrt(A):
    lower = np.linalg.cholesky(A+np.eye(A.shape[1])*(1e-6))
    #squareroot of the determinant of A
    sign, logdet = np.linalg.slogdet(lower)
    return sign*logdet

def single_gauss(x_s, mu, sigma):
    #take the pseudo-inverse only in case the matrix is not invertible
    exponent = -1/2*((x_s-mu).T @ tol_inverse(sigma) @ (x_s-mu))
    #The factor in front of determinant (2*np.pi)**(x_s.shape[0]/2) leads to overflow error
    #for the responsabilities it cancels out anyway
    factor = 1/(log_cholesky_det_sqrt(sigma))
    return factor * np.exp(exponent)

def cost_function(x, mu_2,sigma_2,pi_2, mu_1,sigma_1,pi_1):
    D = sigma_2.shape[1]
    sum_terms_1 = 0
    sum_terms_2 = 0

    inv_sig_1 = [tol_inverse(sigma_1[k]) for k in range(K)]
    log_det_sig_sqrt_1 = [log_cholesky_det_sqrt(sigma_1[k]) for k in range(K)]

    inv_sig_2 = [tol_inverse(sigma_2[k]) for k in range(K)]
    log_det_sig_sqrt_2 = [log_cholesky_det_sqrt(sigma_2[k]) for k in range(K)]
    for s in range(x.shape[0]):
        
        exponents1 = [np.log(pi_1[k])-D/2*np.log(2*np.pi)-log_det_sig_sqrt_1[k]-1/2*((x[s,:]-mu_1[k]).T @ inv_sig_1[k] @ (x[s,:]-mu_1[k])) for k in range(K)]
        exponents2 = [np.log(pi_2[k])-D/2*np.log(2*np.pi)-log_det_sig_sqrt_2[k]-1/2*((x[s,:]-mu_2[k]).T @ inv_sig_2[k] @ (x[s,:]-mu_2[k])) for k in range(K)]
        
        sum_terms_1 += np.max(exponents1) + np.log(np.sum([np.exp(exponents1[k]-np.max(exponents1)) for k in range(K)]))
        sum_terms_2 += np.max(exponents2) + np.log(np.sum([np.exp(exponents2[k]-np.max(exponents2)) for k in range(K)]))
        
    cost = np.abs(sum_terms_1-sum_terms_2)
    
    return cost

def stable_posterior(x,weights,mu, sigma):
    posts = []
    for i in range(len(weights)):

        dim = sigma[0].shape[0]
        exponents = [np.log(weights[k])-dim/2*np.log(2*np.pi)-log_cholesky_det_sqrt(sigma[k])-1/2*((x-mu[k]).T @ tol_inverse(sigma[k]) @ (x-mu[k])) for k in range(K)]
                    
        y_max = np.max(exponents)
        # -> log sum trick: adding the first maximum in front
        log_post_enum = exponents[i]
        log_post_denum = y_max + np.log(np.sum([np.exp(exponents[k]-y_max) for k in range(K)]))
        posts.append(log_post_enum-log_post_denum)
        
    return  [np.exp(post) for post in posts]


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
        ax[0,k].set_title(fr'k = {k}, $\pi_{k}$ = {pi[k]}')

        # Plot the covariance matrix
        ax[1,k].imshow(cov, cmap='viridis', interpolation='nearest')
        # Add a colorbar for covariance matrix
        #plt.gcf().colorbar(im, ax=ax[1,k])

    

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
            # transforming random sample from a standard normal distribution into a sample from our distribution
            # this is essentially the reverse calculation to the z-transform z = (x-mu)/sigma  ->  x = mu +sigma*z
            # the cholesky decomposition of the determinant is like taking its root
            sample_norm = np.random.randn(mu.shape[1])
            lower = np.linalg.cholesky(sigma[k,:,:]+ np.eye(sigma.shape[1])*(1e-6))
            sample = mu[k,:] + lower @ sample_norm
            #########
            M = int(np.sqrt(len(sample)))
            sample_reshaped = sample.reshape(M,M)
            im = ax[j,i].imshow(sample_reshaped, cmap='gray', interpolation='nearest')
            # Add a colorbar for covariance matrix
            #plt.gcf().colorbar(im, ax=ax[0,k])
            ax[j,i].set_title(f'k = {k}')


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
        fig2: 
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
    S = x.shape[0] #2000... number of training images
    M = x.shape[1] #28... width of one image
    D = M*M        #784... number of pixels in an image

    #1.) Flattening data to get proper dimensions
    x = np.reshape(x,(S, M*M))

    #2.) Initialization of covariance matrix and weights for every component
    print("Initialization of sigma and pi ...")
    sigma = [np.eye(D) for k in range(K)]
    pi = np.ones(K)/K
    print("Done \n")

    #3.) Initialization of means with k-means algorithm
    print("Initialization of mu with k-means algorithm ...")
    J = 20
    j = 1
    #distances of every sample to the closest centroid -> initialized as infinity
    distances = np.ones(S)*np.inf
    #centers assigned to each sample image
    center_assignments = np.zeros(S)
    center_assignments_new = np.zeros(S)
    #pick 3 random samples as starting points for the centroids
    centers = [x[np.random.randint(0,S),:] for k in range(K)]

    #stopping criteria 1: max. iterations reached
    while j<=J:
        print(f"Iteration {j}")
        #a) assignment step
        for s in range(S): # iterate over images
            x_s = x[s,:]
            
            for k in range(K): # iterate over clusters and find closest center
                #if distance is smaller than the one previously assigned,
                #assign this as new center
                if np.linalg.norm(x_s-centers[k])<distances[s]:
                    distances[s] = np.linalg.norm(x_s-centers[k])
                    center_assignments_new[s] = k

        #stopping criteria 2: assigned centers didn't change anymore
        if np.any(center_assignments-center_assignments_new):
            center_assignments = np.copy(center_assignments_new)
            j = j+1
        else:
            break

        #b) update step
        #calculating new centroids by taking mean over all assigned samples
        for k in range(K):
            assigned_samples = np.where(center_assignments == k)
            new_center = np.mean(x[assigned_samples[0],:], axis=0)
            centers[k] = new_center

    #centers of the converged k-mean algorithm are used to initialize means of GMM
    mu = centers
    print("Done \n")

    #4.) EM algorithm
    print("Expectaction Maximization algorithm ...")
    
    mu_new = np.copy(mu)
    sigma_new = np.copy(sigma)
    pi_new = np.copy(pi)

    #TODO: adjust max. iteration count if it works after the second iteration
    J = 20
    j = 1
    e_1 = 1e-6

    #first iteration is always executed to get the first updated values for the cost function
    while j<= J:
        print(f"Iteration {j}")

        #pre-calculate inverse of cov matrix and squareroot of the determinant with offset on main diagonal
        #stay the same for one iteration
        inv_sig = [tol_inverse(sigma[k]) for k in range(K)]
        #TODO: In the second iteration, for some reason all the determinants get 0,why?
        log_det_sig_sqrt = [log_cholesky_det_sqrt(sigma[k]) for k in range(K)]

        for k in range(K):
            w_ks_list = []

            #calculating responsabilities
            for s in range(S):
                exponents = [np.log(pi[k])-D/2*np.log(2*np.pi)-log_det_sig_sqrt[k]-1/2*((x[s,:]-mu[k]).T @ inv_sig[k] @ (x[s,:]-mu[k])) for k in range(K)]
                
                y_max = np.max(exponents)
                # -> log sum trick: adding the first maximum in front
                log_w_ks_enum = exponents[k]
                log_w_ks_denum = y_max + np.log(np.sum([np.exp(exponents[k]-y_max) for k in range(K)]))
               
                w_ks = np.exp(log_w_ks_enum-log_w_ks_denum)
                w_ks_list.append(w_ks)

            w_ks_list = np.atleast_1d(np.array(w_ks_list))
            print(w_ks_list.shape)
            print(f"w_s_{k} finished with dim: {w_ks_list.shape}")

            #calculating remaining parameters
            N_k = np.sum(w_ks_list)
            print(f"N_{k} finished")
            mu_new[k] = (w_ks_list @ x)/N_k
            print(f"mu_{k} finished with dim: {mu_new[k].shape}")
            ######
            sigma_new[k] = np.einsum('ij,ik,i->jk', x - mu[k], x - mu[k], w_ks_list) / N_k
            print(f"sigma_{k} finished")
            pi_new[k] = N_k/S
            print(f"pi_{k} finished")


        cost = cost_function(x,mu_new, sigma_new, pi_new,mu,sigma, pi)
        
        #TODO: add the actual cost function once sigma is invertible after the first iteration


        if (cost >= e_1):
            print(f"Remaining cost: {cost} higher than {e_1}")
            mu = np.copy(mu_new)
            sigma = np.copy(sigma_new)
            pi = np.copy(pi_new)

            j = j+1
            
        else:
            print(f"Algorithm converged after {j} iterations.")
            mu = np.copy(mu_new)
            sigma = np.copy(sigma_new)
            pi = np.copy(pi_new)
            break
        print(f"Finished iteration {j-1}")

        #Plotting the GMM components
        ax1 = plot_GMM(mu_new, sigma_new, pi_new, ax1)

        fig1.show()
        fig1.tight_layout()
        fig1.savefig("./Task_2_fig_1.png")

        #drawing random samples
        ax2 = plot_samples(mu_new, sigma_new, pi_new, ax2,num_samples)
        
        fig2.show()
        fig2.tight_layout()
        fig2.savefig("./Task_2_fig_2.png")

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
    
    #TODO: it does something, but probably not the right thing
    mu, sigma, pi = m_params
    K = len(pi)
    M = x.shape[2]
    S, sz, _ = x.shape

    fig, ax = plt.subplots(S, 3, figsize=(3, 8))
    fig.suptitle('Task 3 - Conditional GMM', fontsize=12)
    for a in ax.reshape(-1):
        a.axis('off')

    ax[0, 0].set_title('Condition', fontsize=8), ax[0, 1].set_title('Posterior Exp.', fontsize=8), ax[0, 2].set_title('Groundtruth', fontsize=8)
    for s in range(S):
        ax[s, 2].imshow(x[s], vmin=0, vmax=1., cmap='gray')

    """ Start of your code
    """

    x_2_indices = np.where(mask)[0]
    x_1_indices = np.where([1-n for n in mask])[0]

    for s in range(S):
        x_flat = np.reshape(x[s,:,:], (M * M))
        x_masked_flat = x_flat*mask

        #masked image
        ax[s, 0].imshow(np.reshape(x_flat*mask, (M,M)), vmin=0, vmax=1., cmap='gray')

        x_1 = x_flat[x_1_indices]
        x_2 = x_flat[x_2_indices]

        mu_1 = mu[:,x_1_indices]
        sigma_11 = sigma[:,x_1_indices, :][:,:, x_1_indices]
        mu_2 = mu[:,x_2_indices]
        sigma_22 = sigma[:,x_2_indices, :][:,:, x_2_indices]
        
        sigma_12 = sigma[:,x_1_indices, :][:,:, x_2_indices]
        sigma_21 = sigma[:,x_2_indices, :][:,:, x_1_indices]

        mu_1con2 = [mu_1[k,:] + sigma_12[k,:,:] @tol_inverse(sigma_22[k,:,:]) @ (x_2 -mu_2[k,:]) for k in range(K)]
        sigma_1con2 = [sigma_11[k,:,:] - sigma_12[k,:,:] @ tol_inverse(sigma_22[k,:,:]) @ sigma_21[k,:,:] for k in range(K)]
            
        #pi_1con2_unnormed = [pi[k]*1/(np.sqrt(2*np.pi)*np.exp(log_cholesky_det_sqrt(sigma_22[k,:,:])))*np.exp(-1/2*(x_2-mu_2[k,:]).T @ tol_inverse(sigma_22[k,:,:])@(x_2-mu_2[k,:])) for k in range(K)]
        #pi_1con2 = [val/sum(pi_1con2_unnormed) for val in pi_1con2_unnormed]
        pi_1con2 = stable_posterior(x_2,pi,mu_2,sigma_22)

        #posterior_unnormed = [pi_1con2[k]*1/(np.sqrt(2*np.pi)*np.exp(log_cholesky_det_sqrt(sigma_1con2[k][:,:])))*np.exp(-1/2*(x_1-mu_1con2[k]).T @ tol_inverse(sigma_1con2[k][:,:])@(x_1-mu_1con2[k])) for k in range(K)]
        posteriors = stable_posterior(x_1,pi_1con2,mu_1con2, sigma_1con2)
        #TODO: this is temporary, the caclulation is still unstable and it doesn't make sense to use x_1, wich

        restored_image = np.zeros(M*M)
        #posterior expectation
        restored_image[x_1_indices] = np.dot(pi_1con2,mu_1con2)
        restored_image[x_2_indices] = x_2

        #posterior
        ax[s, 1].imshow(np.reshape(restored_image,(M,M)), vmin=0, vmax=1., cmap='gray')

    
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
    mask = np.random.uniform(0,1,x_train.shape[1]**2)
    mask = [1 if x>=0.9 else 0 for x in mask]
    fig2 = task3(x_test,mask,gmm_params)

    for f in fig1:
        pdf.savefig(f)
    pdf.savefig(fig2)
    pdf.close()
    