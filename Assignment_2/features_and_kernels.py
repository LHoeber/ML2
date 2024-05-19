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

def task1():
    """ Subtask 1: Approximating Kernels

        Requirements for the plot:
        - the first row corresponds to the task in 1.1 and the second row to the task in 1.2

        for each row:
        - the first subplot should contain the Kernel matrix with each entry K_ij for k(x_i,x_j')
        - the second to fifth subplot should contain the corresponding feature approximation when using 1,10,100,1000 features
    """

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(15, 8)
    font = {'fontsize': 18}
    
    feat = ['Fourier', 'Gauss']
    for row in range(2):
        axes[row,4].set_title('Exact kernel', **font)
        axes[row,4].set_xticks([])
        axes[row,4].set_yticks([])
        
        axes[row,0].set_ylabel('%s features' %feat[row], **font)
        for col, R in enumerate([1,10,100,1000]):
            axes[row,col].set_title(r'$\mathbf{Z} \mathbf{Z}^{\top}$, $R=%s$' % R, **font)
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
    
    # generate random 2D data
    N = 1000
    D = 2

    X = np.ones((N,D))
    X[:,0] = np.linspace(-3.,3.,N)
    X[:,1] = np.sort(np.random.randn(N))

    """ Start of your code 
    """
    print(f"Dimension of X: {X.shape}")
    #1)Kernel matrix
    print("\nRandom Fourier features:")
    #1.1) random fourier features
    # exact implementation (left side)
    K_left = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Kij = np.exp(-1/2*(X[i,:] - X[j,:]).T @ (X[i,:] - X[j,:]))
            K_left[i,j] = Kij

    #plotting
    img = axes[0,4].imshow(K_left)
    colorbar = fig.colorbar(img, ax=axes[0,4])

    # approximated implementation (right side) for a certain R
    R_list = [1,10,100,1000]
    K_rights = []
    for m,R in enumerate(R_list):
        print(f"R = {R}")
        K_right = np.zeros((N,N))
        #transforming X with N samples into Z with R samples
        W = np.random.multivariate_normal(np.zeros(D),np.eye(D),R)
        B = [np.random.uniform(0,2*np.pi) for r in range(R)]
        
        # shape of X = N x D
        # shape of Z = N x R
        # shape of W = R x D
        # shape of B = R
        print(f"Dimension of W: {W.shape}")
        print(f"Dimension of B: {np.array(B).shape}")
        Z = np.sqrt(2 / R) * np.cos(X @ W.T + B)
        print(f"Dimension of Z: {Z.shape}")
        K_right = Z.dot(Z.T)
        K_rights.append(K_right)
        
        #plotting
        img = axes[0,m].imshow(K_right)
        colorbar = fig.colorbar(img, ax=axes[0,m])

    #1.2) random gauss features
    print("\nRandom gauss features:")
    # exact implementation (left side)
    K_left = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Kij = np.exp(-1/4*(X[i,:] - X[j,:]).T @ (X[i,:] - X[j,:]))
            K_left[i,j] = Kij

    #plotting
    img = axes[1,4].imshow(K_left)
    colorbar = fig.colorbar(img, ax=axes[1,4])

    # approximated implementation (right side) for a certain R
    R_list = [1,10,100,1000]
    K_rights = []
    for m,R in enumerate(R_list):
        print(f"R = {R}")
        K_right = np.zeros((N,N))
        #transforming X with N samples into Z with R samples
        #TODO: can the same samples be drawn multiple times?
        T = X[np.random.choice(np.arange(0,X.shape[0]), R)]
        sigma = 1
        Z = np.zeros((N, R))
        
        for r in range(R):
            t = T[r]
            prod = -np.linalg.norm((X-t),axis=1)**2
            Z[:, r] = np.sqrt(1 / R) * np.exp(prod/ (2 * sigma**2))
        
        K_right = Z @ Z.T
        K_rights.append(K_right)
        
        #plotting
        img = axes[1,m].imshow(K_right)
        colorbar = fig.colorbar(img, ax=axes[1,m])
        

    """ End of your code 
    """

    return fig

def task2():
    """ Subtask 2: Linear Regression with Feature Transforms

        Requirements for the plot:
        - the left and right subplots should cover the cases with random Fourier and Gauss features, respectively

        for each subplot:
        - plot the averaged (over 5 runs) mean and standard deviation of training and test errors over the number of features
        - include labels for the curves in a legend
    """

    def gen_data(n,d):
        sig = 1. 

        v_star = np.random.randn(d)
        v_star = v_star/np.sqrt((v_star**2).sum())

        # create input data on unit sphere
        x = np.random.randn(n,d)
        x = x/np.sqrt((x**2).sum(1,keepdims=True))
        
        # create targets y
        y = np.zeros((n))
        for n_idx in np.arange(n):
            y[n_idx] = 1/(0.25 + (x[n_idx]).sum()**2) + sig*np.random.randn(1)
        
        return x,y

    n = 200
    n_test = 100
    D = 5

    x_, y_ = gen_data(n+n_test,D)
    idx = np.random.permutation(np.arange(n+n_test))
    x,y,x_test,y_test = x_[idx][:n],y_[idx][:n],x_[idx][n::],y_[idx][n::]

    # features
    R = np.arange(1,100)

    # plot
    fig2, ax = plt.subplots(1,2)
    ax[0].set_title('Random Fourier Features')
    ax[0].set_xlabel('features R')

    ax[1].set_title('Random Gauss Features')
    ax[1].set_xlabel('features R')

    """ Start of your code 
    """
    lam = 0.1

    def z_kernel_fourier(X,R):
        #transforming X with N samples into Z with R samples
        W = np.random.multivariate_normal(np.zeros(D),np.eye(D),R)
        B = [np.random.uniform(0,2*np.pi) for r in range(R)]
        
        Z = np.sqrt(2 / R) * np.cos(X @ W.T + B)
        return Z

    def z_kernel_gauß(X,R):
        N = X.shape[0]
        #transforming X with N samples into Z with R samples
        #TODO: can the same samples be drawn multiple times?
        T = X[np.random.choice(np.arange(0,X.shape[0]), R)]
        sigma = 1
        Z = np.zeros((N, R))
        
        for r in range(R):
            t = T[r]
            prod = -np.linalg.norm((X-t),axis=1)**2
            Z[:, r] = np.sqrt(1 / R) * np.exp(prod/ (2 * sigma**2))
        
        return Z


    def theta_star(x_train,y_train,lam,method,R):
        if method == "fourier":
            z = z_kernel_fourier(x_train,R)

        elif method == "gauss":
            z =  z_kernel_gauß(x_train,R)
            
        else:
            raise NameError(f"'{method}' is not a valid method to calculate the kernel.")
        term_1 = z @ z.T +lam*np.eye(z.shape[0])
        inv_term_1 = np.linalg.pinv(term_1)
        term_2 = inv_term_1.T @ z
        return term_2.T @ np.reshape(y_train,(len(y_train),1))

    def y_hat(x,theta,method):
        R = theta.shape[0]
        if method == "fourier":
            z = z_kernel_fourier(x,R)

        elif method == "gauss":
            z =  z_kernel_gauß(x,R)
        return z.T @ z @ theta
    
    def mse(y,y_pred):
        return np.mean(np.power((y-y_pred),2))

    R_list = [1,2,10,20,50,100]
    runs = 5
    #losses for each R and 5 runs
    fourier_training_loss = np.zeros((runs,len(R_list)))
    gauss_training_loss = np.zeros((runs,len(R_list)))

    fourier_test_loss = np.zeros((runs,len(R_list)))
    gauss_test_loss = np.zeros((runs,len(R_list)))

    for iter in range(runs):
        for r,R in enumerate(R_list):  
            # With random fourier feature transform:
            theta_fourier = theta_star(x,y,lam,"fourier",R)
            y_pred = y_hat(x,theta_fourier,"fourier")
            fourier_training_loss[iter,r] = mse(y,y_pred)

            y_pred = y_hat(x_test,theta_fourier,"fourier")
            fourier_test_loss[iter,r] = mse(y,y_pred)

            # with random gauss feature transform:
            theta_gauss = theta_star(x,y,lam,"gauss",R)
            y_pred = y_hat(x,theta_gauss,"gauss")
            gauss_training_loss[iter,r] = mse(y,y_pred)

            y_pred = y_hat(x_test,theta_gauss,"gauss")
            gauss_test_loss[iter,r] = mse(y,y_pred)

    fourier_training_loss = np.mean(fourier_training_loss,axis = 0)
    gauss_training_loss = np.mean(gauss_training_loss,axis = 0)
    ax[0].plot(fourier_training_loss,label="training set")
    ax[1].plot(gauss_training_loss,label="training set")

    fourier_test_loss = np.mean(fourier_test_loss,axis = 0)
    gauss_test_loss = np.mean(gauss_test_loss,axis = 0)
    ax[0].plot(fourier_test_loss,label="test set")
    ax[1].plot(gauss_test_loss,label="test set")


    """ End of your code 
    """

    ax[0].legend()
    ax[1].legend()

    return fig2

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    fig1 = task1()
    fig2 = task2()
    pdf.savefig(fig1)
    pdf.savefig(fig2)

    pdf.close()


