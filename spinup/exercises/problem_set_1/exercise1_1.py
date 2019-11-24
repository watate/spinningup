import tensorflow as tf
import numpy as np
import math
"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function which takes in Tensorflow symbols for the means and 
log stds of a batch of diagonal Gaussian distributions, along with a 
Tensorflow placeholder for (previously-generated) samples from those 
distributions, and returns a Tensorflow symbol for computing the log 
likelihoods of those samples.

"""

def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################
#My new answer
    presum = -0.5 * (((x - mu)/(tf.exp(log_std)))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(presum, axis = 1)
#My old answer
'''
    # 1. Take in values x, mu, and log_std
    
    batch_size = 32
    dim = 10

    # Calculate std
    std = tf.math.exp(log_std)

    # Set up tf placeholders
    ans = tf.placeholder(tf.float32, shape=[dim,])
    x2 = tf.placeholder(tf.float32, shape=[dim,])
    mu2 = tf.placeholder(tf.float32, shape=[dim,])
    std2 = tf.placeholder(tf.float32, shape=[dim,])
    log_std2 = tf.placeholder(tf.float32, shape=[dim,])
    output = tf.placeholder(tf.float32, shape=[dim,])

    # Set up array for output
    output_np = []

    # 2. Calculate log_likelihood according to spinning up formula
    for i in range(batch_size):
        # intialize variables
        x2 = x[i, :]
        mu2 = mu[i, :]
        std2 = std[:]
        log_std2 = log_std[:]

        # calculate sum portion of likelihood
        A1 = x2 - mu2
        A = tf.pow(A1, 2)
        #test = (x[i, :] - mu[i, :])**2
        #test = tf.keras.backend.print_tensor(test, message = 'test is')
        #A1 = tf.keras.backend.print_tensor(A1, message = 'A1 is')
        #A = tf.keras.backend.print_tensor(A, message = 'A is')
        B1 = tf.pow(std2, 2)
        B = tf.divide(A, B1)
        #B = tf.keras.backend.print_tensor(B, message = 'B is')
        C = B + 2*log_std2
        D = tf.reduce_sum(C)
        E = -0.5*(D + dim * tf.math.log(2*math.pi))
        
        output_np.append(E)
    
    output = output_np
    return 
'''



if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """
    from spinup.exercises.problem_set_1_solutions import exercise1_1_soln
    from spinup.exercises.common import print_result

    sess = tf.Session()

    dim = 10
    x = tf.placeholder(tf.float32, shape=(None, dim))
    mu = tf.placeholder(tf.float32, shape=(None, dim))
    log_std = tf.placeholder(tf.float32, shape=(dim,))

    your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
    true_gaussian_likelihood = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)

    batch_size = 32
    feed_dict = {x: np.random.rand(batch_size, dim),
                 mu: np.random.rand(batch_size, dim),
                 log_std: np.random.rand(dim)}


    your_result, true_result = sess.run([your_gaussian_likelihood, true_gaussian_likelihood],
                                        feed_dict=feed_dict)

    '''
    z = tf.print(your_result)
    z2 = tf.print(true_result)
    sess.run([z, z2])
    '''
    correct = np.allclose(your_result, true_result)
    print_result(correct)