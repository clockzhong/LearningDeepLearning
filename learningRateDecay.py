#! /usr/bin/env python
# this is for problem 4 in assignment 3
# but after I added dynamic learning rate features, the situation becomes worse, why? Is it correct?

batch_size = 128
hidden_size = 2048

def tf_linear(dataset, w,b):
	return tf.matmul(dataset,w) + b

dropout = 0.5 # Dropout, probability to keep units

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_size]))
  biases1 = tf.Variable(tf.zeros([hidden_size]))
  weights2 = tf.Variable(
    tf.truncated_normal([hidden_size, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  #first layer:
  logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
  #ReLU
  relu=tf.nn.relu(logits1)

  #add Dropout
  keep_prob = tf.placeholder(tf.float32)
  relu_drop = tf.nn.dropout(relu, keep_prob)

  #Second layer:
  logits2 = tf.matmul(relu_drop, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2))
  beta=0.0005
  loss+=beta*(tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
 

  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.5, global_step,  100000, 0.96, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits2)
  valid_prediction = tf.nn.softmax(
    tf_linear(tf.nn.relu(tf_linear(tf_valid_dataset, 
                                             weights1, biases1)),
                                             weights2, biases2)
                                             )
  test_prediction = tf.nn.softmax(
    tf_linear(tf.nn.relu(tf_linear(tf_test_dataset, 
                                             weights1, biases1)),
                                             weights2, biases2)
                                             )

num_steps = 12001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: dropout}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

