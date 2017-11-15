# imports
import numpy as np
import tensorflow as tf

# modules
from helpers import load_json, load_glove_model, tokenize_and_map, get_labels, batch_iter
from models import SimpleGRU

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_units', 64, 'Number of units in LSTM layers')
flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers')
flags.DEFINE_integer('num_classes', 4, 'Number of output classes')
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs for training.')

# file paths
json_path = "data/test.jsonl"
glove_path = "glove/glove.6B.50d.txt"
model_path = "./model_params/model_final"

def run_evaluation(word_vectors, ids_matrix, indices):

    num_dimensions = word_vectors.shape[1]
    max_length = ids_matrix.shape[1]

    with tf.Graph().as_default():

        # Define data placeholders
        data = tf.placeholder(tf.int32, [None, max_length])
        labels = tf.placeholder(tf.int64, [None])

        # Init model and crate ops, to make sure that all variables are properly initialized by the initializer
        model = SimpleGRU(word_vectors, FLAGS.num_layers, num_dimensions, max_length, FLAGS.num_units, FLAGS.num_classes, data, labels)
        eval_op = model.accuracy
        pred_op = model.predicted_labels

        saver = tf.train.Saver()

        predictions = []
        batch_accuracies = []
        with tf.Session() as sess:
            saver.restore(sess, model_path)

            step_count = 0
            for batch_x, batch_y in batch_iter(ids_matrix, indices, FLAGS.batch_size, FLAGS.epochs, shuffle=False):
                [pred, acc] = sess.run([pred_op, eval_op], feed_dict={data:batch_x, labels:batch_y})
                predictions.append(pred)
                batch_accuracies.append(acc)
                if step_count % 5 == 0:
                    print("Step %d" % (step_count))
                step_count += 1

        return predictions, batch_accuracies


def main():

    # Load Data
    datastore = load_json(json_path)
    word_vectors, words_list = load_glove_model(glove_path)
    ids_matix = tokenize_and_map(datastore, words_list)
    indices = get_labels(datastore)

    # Generate predictions
    predictions, batch_accuracies = run_evaluation(word_vectors, ids_matix, indices)

    # Display accuracy
    print (np.average(batch_accuracies))

    # Save predictions
    np.save("predictions", np.concatenate(predictions))


if __name__ == "__main__":
    main()


