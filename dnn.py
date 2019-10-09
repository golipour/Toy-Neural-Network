import numpy as np
import math
import copy

epsilon = 1e-15

### DNN ARCHITECTURE
#layer = [(X.shape[0], None), (20, relue), (7, relue), (5, relue), (1, sigmoid)]

### ACTIVATION FUNCTIONS AND THEIR DERIVATIVES ###
def relue(z):
    a = np.maximum(z, 0)
    return a

def d_relue(l):
    da2dz = np.ones(Z[l].shape)
    da2dz[ Z[l] <= 0] = 0
    return da2dz

def sigmoid(z):
    a = 1. / ( 1. + np.exp(-z))
    return a

def d_sigmoid(l):
    da2dz = A[l] * (1 - A[l])
    return da2dz

def tanh(z):
    a = np.tanh(z)
    return a

def d_tanh(l):
    da2dz = 1 - A[l]**2
    return da2dz

def derivative(func):
    d_func = func.copy()
    for idx, f in enumerate(func):
        if f is relue:
            d_func[idx] = d_relue
        elif f is sigmoid:
            d_func[idx] = d_sigmoid
        elif f is tanh:
            d_func[idx] = d_tanh
        else:
            d_func[idx] = None
    return d_func
### ACTIVATION FUNCTIONS AND THEIR DERIVATIVES ###


### LOSS FUNCTIONS AND THEIR DERIVATIVES ###
def loss_kl_binary(A, Y):
    zero_elements = (A == 0)
    #assert(zero_elements.any())
    A[zero_elements] += epsilon
    one_elements = (A == 1)
    A[one_elements] -= epsilon
    
    loss = - ( Y * np.log(A) + (1 - Y) * np.log(1 - A) )
    return loss

def d_loss_kl_binary(A, Y):
    zero_elements = (A == 0)
    #assert(zero_elements.any())
    A[zero_elements] += epsilon
    one_elements = (A == 1)
    A[one_elements] -= epsilon
    
    dLoss2dA = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    return dLoss2dA

#def loss(Y_hat, Y):
#    loss = - np.su(Y * np.log(Y_hat), axis=0, keepdims=True) 
#    return loss

#def d_loss(Y_hat, Y):
#    dLoss2dY_hat = - np.divide(Y, Y_hat)

### LOSS FUNCTIONS AND THEIR DERIVATIVES ###


### REGULATOR FUNCTIONS AND THEIR DERIVATIVES ###
def reg(W):
    global lambdaa, num_samples
    r = 0
    for w in W[1:]:
        r += np.sum(np.square(w))
        
    r *= (lambdaa / num_samples * 2)
    return r

def d_reg(Wl):
    global lambdaa, num_samples
    dReg2dWl = (lambdaa / num_samples) * Wl
    return dReg2dWl
### REGULATOR FUNCTIONS AND THEIR DERIVATIVES ###


### COST FUNCTIONS AND THEIR DERIVATIVES ###
def cost(A, Y):
    m = Y.shape[1]   
    cost = (1. / m) * np.sum( loss(A, Y))
    return cost

def d_cost(A, Y):
    m = Y.shape[1]   
    dCost2dA = (1./m) * d_loss(A, Y)
    return dCost2dA

def cost_reg(A, Y):
    global W
    m = Y.shape[1]   
    cost = (1. / m) * np.sum(loss(A, Y)) + reg(W)
    return cost
### COST FUNCTIONS AND THEIR DERIVATIVES ###

### DROP_OUT ###
def random_drop_mask():
    global keep_prob
    D = []
    for l in range(0, L):
        keep_mask = np.random.rand(num_units[l], A[0].shape[1])
        keep_mask =  (keep_mask < keep_prob[l])
        D.append(keep_mask)
    return D
    
def drop_out(l, drop_mask_l):
    global A, keep_prob
    A[l] *= drop_mask_l 
    A[l] /= keep_prob[l]          
### DROP_OUT ###

### UTILITIES ###
def error(Y_, Y):
    m = Y.shape[1]
    err = np.sum(Y_ != Y) / m
    return err

def accuracy(Y_, Y):
    acc = 1 - error(Y_, Y)
    #m = Y.shape[1]
    #acc = np.sum(Y_ == Y) / m
    return acc

def label(probs):
    labels = np.zeros(probs.shape)
    labels[probs >= .5] = 1
    return labels

def normalize(X):
    X_ = X - np.mean(X, axis=1, keepdims=True)
    X_ /= np.std(X_, axis=1, keepdims=True)
    return X_

def rand(interval, num_samples):
    a = interval[0]
    b = interval[1]
    samples = (b - a) * np.random.random_sample((num_samples, )) + a
    return samples

def rand_log(interval, num_samples):
    a = np.log10(interval[0] + epsilon)
    b = np.log10(interval[1] + epsilon)
    samples = (b - a) * np.random.random_sample((num_samples, )) + a
    samples = np.power(10, samples)
    return samples
### UTILITIES ###


### BATCH NORMALIZATION ###
def batch_norm_forward(Z_l, l, update = False):
    #global miu_batch, var_batch
    miu_batch[l] = np.mean(Z_l, axis=1, keepdims=True)
    var_batch[l] = np.var(Z_l, axis=1, keepdims=True)
    Z_norm[l] = (Z_l - miu_batch[l]) / np.sqrt(var_batch[l] + epsilon)
    Z_tilde_l = gamma[l] * Z_norm[l] + beta[l]
    if update:
        miu[l] = .9 * miu[l] + .1 * miu_batch[l]
        var[l] = .9 * var[l] + .1 * var_batch[l]
        
    return Z_tilde_l

def batch_norm_backward(dZ_tilde_l, l):
    m = dZ_tilde_l.shape[1]
    sigma_inv = 1. / np.sqrt(var[l] + epsilon)
    dZ_norm_l = dZ_tilde[l] * gamma[l]
    dvar_l = np.sum(dZ_norm_l * (Z[l] - miu_batch[l]), axis=1, keepdims = True ) * -0.5 * sigma_inv**3
    dmiu_l = np.sum(dZ_norm_l * -sigma_inv, axis=1, keepdims = True)
    dbeta_l = np.sum(dZ_tilde[l], axis = 1, keepdims = True)   # dJ/dbeta[l] = dJ/dZ[l] * dZ[l]/dbeta[l] (sum over samples)
    dgamma_l = np.sum(dZ_tilde[l] * Z_norm[l], axis = 1, keepdims = True)
    dZ_l = dZ_norm_l * sigma_inv + dvar_l * (2./m) * (Z[l] - miu_batch[l]) + dmiu_l * (1./m)            
    
    return dZ_l, dgamma_l, dbeta_l
### BATCH NORMALIZATION ###


### MINI-BATCH ###
def shuffle(data_set):
    X_ = data_set[0]
    Y_ = data_set[1]
    m = Y_.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X_[:, permutation]
    shuffled_Y = Y_[:, permutation].reshape((1,m))
    shuffled_data_set = (shuffled_X, shuffled_Y)
    return shuffled_data_set

def gen_mini_batches(data_set, mini_batch_size = 64):
    X_ = data_set[0]
    Y_ = data_set[1]
    m = Y_.shape[1]                
    mini_batches = []
    
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = X_[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = Y_[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = X_[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = Y_[:, num_complete_minibatches*mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches(data_set, mini_batch_size = 64):
    m = data_set[1].shape[1]
    if mini_batch_size == m:                      # just for less computation
        batch = [data_set]
    else:
        shuffled_data_set = shuffle(data_set)
        batch = gen_mini_batches(shuffled_data_set, mini_batch_size)
    return batch
### MINI-BATCH ###


### INITIALIZING MODEL ###
def init_model(train_set = (None, None), dev_set = (None, None), layers = [], BN = False, reg_term = 0, dropout_probs = [], initialization = "random"):
    global X, Y, X_dev, Y_dev, A, num_samples
    global layer, L, num_units, activation_func, d_activation_func
    global lambdaa
    global drop, keep_prob, drop_mask
    global W, b, Z, A
    global dW, db, dZ, dA
    global forward_propagate_, backward_propagate_
    global loss, d_loss
    global batch_normalized, gamma, dgamma, beta, dbeta, miu, var, miu_batch, var_batch, Z_norm, Z_tilde, dZ_tilde
    
    layer = layers
    L = len(layer)                            # input layer = 0, output layer = L-1
    num_units = [l[0] for l in layer]         # number of units at each layer
    activation_func = [l[1] for l in layer]
    d_activation_func = derivative(activation_func)
    
    loss = loss_kl_binary
    d_loss = d_loss_kl_binary
    
    lambdaa = reg_term

    
    if len(dropout_probs)==L:
        drop = True
        keep_prob = dropout_probs
        forward_propagate_ = forward_propagate_drop
        backward_propagate_ = backward_propagate_drop
    else:
        drop = False
        forward_propagate_ = forward_propagate
        backward_propagate_ = backward_propagate
    
    W  = [None]*L
    dW = [None]*L
    b  = [None]*L
    db = [None]*L
    
    Z  = [None]*L
    A  = [None]*L
    dZ = [None]*L
    dA = [None]*L
    
    if BN:
        batch_normalized = True
        gamma = [None]*L
        beta = [None]*L
        dgamma = [None]*L
        dbeta = [None]*L
        miu = [None]*L
        var = [None]*L
        miu_batch = [None]*L
        var_batch = [None]*L
        Z_norm = [None]*L
        Z_tilde = [None]*L
        dZ_tilde = [None]*L
        forward_propagate_ = forward_propagate_bn
        backward_propagate_ = backward_propagate_bn
    else:
        batch_normalized = False
    
    X = train_set[0]
    Y = train_set[1]
    A[0] = X
    num_samples = X.shape[1]
    
    X_dev = dev_set[0]
    Y_dev = dev_set[1]
    
    init_parameters(initialization)
### INITIALIZING MODEL ###


### INITIALIZING PARAMETERS ###
def init_parameters(initialization):
    # Initialize parameters dictionary.
    if initialization == "zeros":
        init_parameters_zeros()
    elif initialization == "random":
        init_parameters_random()
    elif initialization == "xavier":
        init_parameters_xavier()
    elif initialization == "he":
        init_parameters_he()
    
    if batch_normalized:
        init_parameters_bn()

def init_parameters_zeros():
    for l in range(1, L):  #for 1 to L-1
        W[l] = np.zeros((num_units[l], num_units[l-1]))
        b[l] = np.zeros((num_units[l], 1))

def init_parameters_random():
    for l in range(1, L):  #for 1 to L-1
        W[l] = np.random.randn(num_units[l], num_units[l-1]) 
        b[l] = np.zeros((num_units[l], 1))

def init_parameters_xavier():
    for l in range(1, L):  #for 1 to L-1
        W[l] = np.random.randn(num_units[l], num_units[l-1]) * np.sqrt(1. / num_units[l-1])
        b[l] = np.zeros((num_units[l], 1))

def init_parameters_he():
    for l in range(1, L):  #for 1 to L-1
        W[l] = np.random.randn(num_units[l], num_units[l-1]) * np.sqrt(2. / num_units[l-1])
        b[l] = np.zeros((num_units[l], 1))
        
def init_parameters_bn():
    for l in range(1, L):  #for 1 to L-1
        gamma[l] = np.ones((num_units[l], 1))
        #gamma[l] = np.random.randn(num_units[l], 1)
        beta[l] = np.zeros((num_units[l], 1))
        miu[l] = np.zeros((num_units[l], 1))
        var[l] = np.zeros((num_units[l], 1))
### INITIALIZING PARAMETERS ###

def forward_propagate():
    for l in range(1, L):
        Z[l] = W[l] @ A[l-1] + b[l]
        A[l] = activation_func[l](Z[l])

def backward_propagate():
    dA[-1] = d_cost(A[-1], Y_mini_batch)                   # dJ/dA[-1]
    for l in range(L-1,0,-1):
        dZ[l] = dA[l] *  d_activation_func[l](l)           # dJ/dZ[l] = dJ/dA[l] * dA[l]/dZ[l]
        dW[l] = dZ[l] @ A[l-1].T  + d_reg(W[l])            # dJ/dW[l] = dJ/dZ[l] * dZ[l]/dA[l-1] + dReg/dW[l]
        db[l] = np.sum(dZ[l], axis = 1, keepdims = True)   # dJ/db[l] = dJ/dZ[l] * dZ[l]/db   (sum = muliply by ones)
        dA[l-1] = W[l].T @ dZ[l]                           # dJ/dA[l-1]

def forward_propagate_bn(update_miu_sig = True):
    for l in range(1, L):
        Z[l] = W[l] @ A[l-1]
        Z_tilde[l] = batch_norm_forward(Z[l], l, update_miu_sig)
        A[l] = activation_func[l](Z_tilde[l])

def backward_propagate_bn():
    dA[-1] = d_cost(A[-1], Y_mini_batch)                            # dJ/dA[-1]
    for l in range(L-1,0,-1):
        dZ_tilde[l] = dA[l] *  d_activation_func[l](l)              # dJ/dZ_tilde[l] = dJ/dA[l] * dA[l]/dZ_tilde[l]
        dZ[l], dgamma[l], dbeta[l] = batch_norm_backward(dZ_tilde[l], l)
        dA[l-1] = W[l].T @ dZ[l]                                    # dJ/dA[l-1] 
        dW[l] = dZ[l] @ A[l-1].T  + d_reg(W[l])                     # dJ/dW[l] = dJ/dZ[l] * dZ[l]/dA[l-1] + dReg/dW[l]
        

def forward_propagate_drop():
    global drop_mask
    drop_mask = random_drop_mask()
    for l in range(1, L):
        drop_out(l-1, drop_mask[l-1])
        Z[l] = W[l] @ A[l-1] + b[l]
        A[l] = activation_func[l](Z[l])

def backward_propagate_drop():
    dA[-1] = d_cost(A[-1], Y_mini_batch)                   # dJ/dA[-1]
    for l in range(L-1,0,-1):                              # L-1 = output layer
        drop_out(l, drop_mask[l])
        dZ[l] = dA[l] *  d_activation_func[l](l)           # dJ/dZ[l] = dJ/dA[l] * dA[l]/dZ[l]
        dW[l] = dZ[l] @ A[l-1].T  + d_reg(W[l])            # dJ/dW[l] = dJ/dZ[l] * dZ[l]/dA[l-1] + dReg/dW[l]
        db[l] = np.sum(dZ[l], axis = 1, keepdims = True)   # dJ/db[l]
        dA[l-1] = W[l].T @ dZ[l]                           # dJ/dA[l-1]

### UPDATE PARAMETERS ####
def update_parameters_gd():
    for l in range(1, L):
        W[l] = W[l] - learning_rate * dW[l]
        b[l] = b[l] - learning_rate * db[l]

def update_parameters_momentum():
    global t
    t += 1
    for l in range(1, L):
        VdW[l] = beta0 * VdW[l] + (1 - beta0) * dW[l]
        Vdb[l] = beta0 * Vdb[l] + (1 - beta0) * db[l]
        VdW_unbias = VdW[l] / (1 - np.power(beta0, t))
        Vdb_unbias = Vdb[l] / (1 - np.power(beta0, t))
        
        W[l] = W[l] - learning_rate * VdW_unbias
        b[l] = b[l] - learning_rate * Vdb_unbias

def update_parameters_adam():
    global t
    t += 1
    for l in range(1, L):
        VdW[l] = beta1 * VdW[l] + (1 - beta1) * dW[l]
        Vdb[l] = beta1 * Vdb[l] + (1 - beta1) * db[l]
        VdW_unbias = VdW[l] / (1 - np.power(beta1, t))
        Vdb_unbias = Vdb[l] / (1 - np.power(beta1, t))
        
        SdW[l] = beta2 * SdW[l] + (1-beta2) * np.power(dW[l], 2)
        Sdb[l] = beta2 * Sdb[l] + (1-beta2) * np.power(db[l], 2)
        SdW_unbias = SdW[l] / (1 - np.power(beta2, t))
        Sdb_unbias = Sdb[l] / (1 - np.power(beta2, t))
        
        W[l] = W[l] - learning_rate * VdW_unbias / np.sqrt(SdW_unbias + epsilon)
        b[l] = b[l] - learning_rate * Vdb_unbias / np.sqrt(Sdb_unbias + epsilon)        
### UPDATE PARAMETERS ####  

def update_parameters_bn_gd():
    for l in range(1, L):
        W[l] = W[l] - learning_rate * dW[l]
        beta[l] = beta[l] - learning_rate * dbeta[l]
        gamma[l] = gamma[l] - learning_rate * dgamma[l]

### OPTIMIZER ###
def init_gd(optimizer):
    global mini_batch_size, learning_rate
    mini_batch_size = optimizer['mini_batch_size']
    learning_rate = optimizer['learning_rate']
    
def init_momentum(optimizer):
    global mini_batch_size, learning_rate, beta0
    global VdW, Vdb, t
    mini_batch_size = optimizer['mini_batch_size']
    learning_rate = optimizer['learning_rate']
    beta0 = optimizer['beta0']
    
    t = 0
    VdW = [None]*L
    Vdb = [None]*L
    for l in range(1, L):  #for 1 to L-1
        VdW[l] = np.zeros(W[l].shape)
        Vdb[l] = np.zeros(b[l].shape)

def init_adam(optimizer):
    global mini_batch_size, learning_rate, beta1, beta2
    global VdW, Vdb, SdW, Sdb, t
    mini_batch_size = optimizer['mini_batch_size']
    learning_rate = optimizer['learning_rate']
    beta1 = optimizer['beta1']
    beta2 = optimizer['beta2']
    
    t = 0
    VdW = [None]*L
    Vdb = [None]*L
    SdW = [None]*L
    Sdb = [None]*L
    for l in range(1, L):  #for 1 to L-1
        VdW[l] = np.zeros(W[l].shape)
        Vdb[l] = np.zeros(b[l].shape)
        SdW[l] = np.zeros(W[l].shape)
        Sdb[l] = np.zeros(b[l].shape)

def init_optimizer(optimizer):
    global update_parameters
    if optimizer['type'] == "gd":
        init_gd(optimizer)
        update_parameters = update_parameters_gd
    elif optimizer['type'] == "momentum":
        init_momentum(optimizer)
        update_parameters = update_parameters_momentum
    elif optimizer['type'] == "adam":
        init_adam(optimizer)  
        update_parameters = update_parameters_adam
    
    if batch_normalized:
        update_parameters = update_parameters_bn_gd
### OPTIMIZER ###


### GRADIENT DESENT ITTERATION ###
def itterate(num_epochs = 1000, optimizer = {}, print_step = 0):
    global A, X_mini_batch, Y_mini_batch
    global cost_train, cost_dev, err_train, err_dev
    
    init_optimizer(optimizer)
    
    batch = random_mini_batches((X, Y), mini_batch_size)
    num_mini_batches = len(batch)
    
    cost_train = np.zeros(num_epochs * num_mini_batches)
    err_train = np.zeros(num_epochs * num_mini_batches)
    cost_dev = np.zeros(num_epochs * num_mini_batches)
    err_dev = np.zeros(num_epochs * num_mini_batches)
    
    j = 0
    for i in range(num_epochs):    
        batch = random_mini_batches((X, Y), mini_batch_size)        
        
        for mini_batch in batch:
            X_mini_batch = mini_batch[0]
            Y_mini_batch = mini_batch[1]
            
            A[0] = X_mini_batch
            forward_propagate_()
            backward_propagate_()
            update_parameters()

            compute_objectives_train(j)
            #if dev_set != None: compute_objectives_dev(j)
            j+=1
            
        if print_step and (i % print_step == 0):
            display(i, j-1)
    
    if print_step:
        display(i, j-1)
### GRADIENT DESENT ITTERATION ###

### PREDICT ###
def predict(X_, recover_AZ = True):
    
    global A, Z
    if recover_AZ:
        A_temp = copy.deepcopy(A)
        Z_temp = copy.deepcopy(Z)
    
    # Forward propagation
    A[0] = X_
    forward_propagate()
    probs = A[-1]
    labels = label(probs)
    
    # recovering A and Z 
    if recover_AZ:
        A = A_temp
        Z = Z_temp
        
    return probs, labels
### PREDICT ###

def compute_objectives_train(i):
    cost_train[i] = cost(A[-1], Y_mini_batch)                # unregularized cost
    err_train[i] = error(label(A[-1]), Y_mini_batch)

def compute_objectives_dev(i):
    global A
    
    A[0] = X_dev
    forward_propagate()
    cost_dev[i] = cost(A[-1], Y_dev)          # unregularized cost
    err_dev[i] = error(label(A[-1]), Y_dev)

        
### DISPLAY ###
def display(ep, it):
    print ("epoch {:>5}{:<5}cost_train:{:.2f} cost_dev:{:.2f} | err_train:{:.2f} err_dev:{:.2f} ".format(ep, ':', cost_train[it], cost_dev[it], err_train[it], err_dev[it]));
### DISPLAY ###
