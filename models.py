"""Builds the classification network.

Implements the inference/loss/optimize pattern for model building.
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. optimize() - Adds to the loss model the Ops required to generate and
apply gradients.


"""

import tensorflow as tf


class SimpleGRU:
    def __init__(self, word_vectors, num_layers, num_dimensions, max_length, num_units, num_classes, input_data,
                 indices):

        self.word_vectors = tf.constant(word_vectors)
        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.max_length = max_length
        self.num_units = num_units
        self.num_classes = num_classes
        self.input_data = input_data
        self.indices = indices

        self._inference = None
        self._loss = None
        self._optimize = None
        self._accuracy = None
        self._predicted_labels = None

    @staticmethod
    def length(sequence):
        """
        Fuction that will return the length of a given indices sequence on the fly.
        :param sequence: Tensor that conatins a batch of sequences
        :return: Array of lengths
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        """
        Function that will return the relevant output of the LSTM cell given the length of the
        input sequence
        :param output: Output tensor of LSTM
        :param length: Array of lengths
        :return: Tensor with relevant outputs
        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @property
    def inference(self):
        if self._inference is None:
            input_dim = tf.shape(self.input_data)
            # data = tf.Variable(tf.zeros([input_dim[0], input_dim[1], self.num_dimensions]), dtype=tf.float32)
            data = tf.nn.embedding_lookup(self.word_vectors, self.input_data)

            cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(num_units=self.num_units)
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.75)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

            data_lenght = self.length(data)

            output, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float64,
                sequence_length=data_lenght,
                inputs=data)

            last = self.last_relevant(output, data_lenght)

            #fully_connected = tf.layers.dense(inputs=last, units=32, activation=tf.nn.sigmoid)
            self._inference = tf.layers.dense(inputs=last, units=self.num_classes, name="inference")
        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            self._loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.inference, labels=self.indices), name="loss")
        return self._loss

    @property
    def optimize(self):
        if self._optimize is None:
            self._optimize = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.loss, name="optimizer")
        return self._optimize

    @property
    def predicted_labels(self):
        if self._predicted_labels is None:
            pred = tf.nn.softmax(self.inference)
            self._predicted_labels = tf.argmax(pred, 1)
        return self._predicted_labels

    @property
    def accuracy(self):
        if self._accuracy is None:
            mistakes = tf.not_equal(self.predicted_labels, self.indices)
            self._accuracy = 1 - tf.reduce_mean(tf.cast(mistakes, tf.float64))
        return self._accuracy
