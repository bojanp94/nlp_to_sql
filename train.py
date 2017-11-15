# imports
import numpy as np
import tensorflow as tf
import argparse
import os

# modules
from helpers import load_json, load_glove_model, tokenize_and_map, get_labels, batch_iter
from models import SimpleGRU

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_units', 64, 'Number of units in LSTM layers')
flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers')
flags.DEFINE_integer('num_classes', 4, 'Number of output classes')
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_integer('epochs', 2, 'Number of epochs for training.')

# file paths
json_path = "data/train.jsonl"
glove_path = "glove/glove.6B.50d.txt"
file_writer = "board/board_2_layers"
model_path = "model_params/model_2_layers"


def run_training(word_vectors, ids_matrix, indices):
    num_dimensions = word_vectors.shape[1]
    max_length = ids_matrix.shape[1]

    with tf.Graph().as_default():

        # Define data placeholders
        data = tf.placeholder(tf.int32, [None, max_length])
        labels = tf.placeholder(tf.int64, [None])

        # Init model and crate ops, to make sure that all variables are properly initialized by the initializer
        model = SimpleGRU(word_vectors, FLAGS.num_layers, num_dimensions, max_length, FLAGS.num_units,
                          FLAGS.num_classes, data, labels)
        train_op = model.optimize
        eval_op = model.accuracy

        # Create summaries for the tensorboard
        tf.summary.scalar("Loss", model.loss)
        tf.summary.scalar("Accuracy", model.accuracy)
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(file_writer)
            writer.add_graph(sess.graph)

            step_count = 0
            for batch_x, batch_y in batch_iter(ids_matrix, indices, FLAGS.batch_size, FLAGS.epochs, shuffle=True):
                [_, train_loss] = sess.run([train_op, model.loss], feed_dict={data: batch_x, labels: batch_y})

                if step_count % 5 == 0:
                    train_accuracy = sess.run(eval_op, feed_dict={data: batch_x, labels: batch_y})
                    print("Step %d, train accuracy %g train loss %g" % (step_count, train_accuracy, train_loss))

                    summary_str = sess.run(summary, feed_dict={data: batch_x, labels: batch_y})
                    writer.add_summary(summary_str, step_count)
                    writer.flush()
                step_count += 1

                saver.save(sess, model_path)


def main():

    # Load Data
    datastore = load_json(json_path)
    word_vectors, words_list = load_glove_model(glove_path)
    ids_matix = tokenize_and_map(datastore, words_list)
    indices = get_labels(datastore)

    # Train and save model
    run_training(word_vectors, ids_matix, indices)

if __name__ == "__main__":
    main()

