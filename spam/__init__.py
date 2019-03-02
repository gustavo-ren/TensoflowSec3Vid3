# Very simple house price predictor who takes only the house size
# implemented in Tensorflow


import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation # import animation support

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# generation of house sizes
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate house prices from house size
np.random.seed(42)
house_price = house_size*100.0+np.random.randint(low=20000, high=70000, size=num_house)


# Plotting the generated data
plt.plot(house_size, house_price, "bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()


# To prevent under/overflows we need to assure the data is on a similar scale
# This step is data preparation
def normalize(array):
    return (array - array.mean())/array.std()


# Define number of training samples(70%), can take the first 70% since the data is random
num_train_samples = math.floor(num_house*0.7)

# Define training data
train_house_size = np.asarray(house_size[:num_train_samples:])
train_house_price = np.asarray(house_price[:num_train_samples:])

# Normalizing the training data
train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[:num_train_samples])

# Normalizing the test data
test_house_price_norm = normalize(test_house_price)
test_house_size_norm = normalize(test_house_size)

"""
    Set TensorFlow placeholders:
        placeholder are tensors that are fed data when the
        graph is executed    
"""
tf_house_size = tf.placeholder("float", name="house_size")  # the name attribute sets the name of the tensor operation
tf_house_price = tf.placeholder("float", name="price")  # in the computation graph.

"""Variables holding the siz_factor and price during training
   they are initialized to some random values based on the normal
   distribution
"""
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# Cost function - Mean Squared Error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_house_price, 2))/(2*num_train_samples)

# Learning rate
learning_rate = 0.1

# Minimize loss by Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)
    # Set how often to display the training progress and the number of training iterations
    display_every = 2
    num_training_iter = 50

    # Keep iterating the training data
    for iteration in range(num_training_iter):
        for (x, y) in zip(train_house_size_norm, train_house_price_norm):  # Fit all training data
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_house_price: y})

        # Display current status
        if (iteration+1) % display_every ==0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
            print("iteraion #:", "%04d"%(iteration+1), "cost=", "{:.9f}".format(c),
                  "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Optimization finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
    print("Trained cost=", training_cost, "size_facto=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("price")
    plt.xlabel("Size(sq.ft)")
    plt.plot(train_house_size, train_house_price, "go", label="Training data")
    plt.plot(train_house_size_norm * train_house_size.std() + train_house_size.mean(),
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset))
             * train_house_price.std() + train_house_price.mean(), label="Learned Regression")

    plt.show()

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_house_price_mean = train_house_price.mean()
    train_house_price_std = train_house_price.std()
