#Imports
import json
import numpy as np
import re

#Constants
STRIP_SPECIAL_CHARS = re.compile("[^A-Za-z0-9 ]+")

#Functions
def load_json(file_path):
    """
    Wrapper function that uses the python json library to load a file.
    :param file_path: Sting that contains the file path of the json file.
    :return: Dictionary with the data form the json file.
    """
    print ("Loading Json File")
    datastore = []
    for line in open(file_path, 'r'):
        datastore.append(json.loads(line))
    print ("Json loading done")
    return datastore


def load_glove_model(glove_file):
    """
    Loads pertained word embeddings. Unknown words are represented with a zeros vector
    :param glove_file: Sting that contains the file path of the glove file.
    :return: Numpy array with the vector embeddings and ordered word list
    """
    print ("Loading Glove Model")
    f = open(glove_file,'r')
    words_list = []
    words_list.append("") #Add an empty string
    words_list.append("Unknown words") # Add a representation for new words
    word_vectors = []
    word_vectors.append(np.zeros(50)) #Add an zeros vector for the empty string
    word_vectors.append(np.zeros(50)) #Add an zeros vector for the empty string
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        words_list.append(word)
        embedding = np.array([float(val) for val in splitLine[1:]])
        word_vectors.append(embedding)
    print ("Glove loading done")
    return np.stack(word_vectors), words_list

def clean_question(string):
    """
    Cleans questions of special characters and converts them to lower case.
    :param string: The string to be cleaned
    :return: Cleaned string
    """
    string = string.lower().replace("<br />", " ")
    return re.sub(STRIP_SPECIAL_CHARS, "", string.lower())

def pad_to_dense(list_of_arrays):
    """
     Appends the minimal required amount of zeroes at the end of each
     array in the list and returns a square matrix
    :param list_of_arrays: List that contains numpy arrays of possibly different length.
    :return: Square matrix
    """
    max_len = max(len(r) for r in list_of_arrays)
    padded = np.zeros((len(list_of_arrays), max_len), dtype=np.int32)
    for enu, row in enumerate(list_of_arrays):
        padded[enu, :len(row)] += row
    return padded

def tokenize_and_map(datastore, words_list):
    """
    Clean, and tokenize all questions in the datastore and subsequently map them to the word embedings indices
    :param datastore: A dictionary that contains the data form the supplied json files
    :param words_list: An ordered list that contains the words of the Glove Model
    :return: Square numpy matrix, where each line represents the word indices for the appropriate question
    """
    ids_list = []
    for sample in datastore:
        if sample["sql"]['agg'] >= 0 and sample["sql"]['agg'] <= 3:
            question = sample['question']
            cleaned_question = clean_question(question)
            split = cleaned_question.split()
            index_array = np.zeros(len(split), dtype=np.int32)  # Create an array with the number of words in the question
            index_counter = 0
            for word in split:
                try:
                    index_array[index_counter] = words_list.index(word)
                except ValueError:
                    index_array[index_counter] = words_list.index("Unknown words")
                index_counter += 1
            ids_list.append(index_array)

    return pad_to_dense(ids_list)

def get_labels(datastore):
    """
    Create a list of labels form the supplied datastore
    :param datastore: A dictionary that contains the data form the supplied json files
    :return: List of integers corresponding to the type of aggregate fuction
    """
    labels_list = []
    for sample in datastore:
        if sample["sql"]['agg'] >= 0 and sample["sql"]['agg'] <= 3:
            labels_list.append(sample["sql"]['agg'])

    return np.stack(labels_list)

def batch_iter(x, y, batch_size, epochs, shuffle):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'x' and the input data 'y')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `x` and `y`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_x, minibatch_y in batch_iter(x, y, 32):
        <DO-SOMETHING>
    :param tx: input data
    :param y: labels
    :param batch_size: the desired size of the batch
    :param epochs: the number of epochs (repetitions of the input data)
    :return: yields mini-batches
    """
    data_size = len(y)
    num_batches = int(np.ceil(data_size / batch_size))

    for epoch in range(epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = x[shuffle_indices]
            shuffled_y = y[shuffle_indices]
        else:
            shuffled_x = x
            shuffled_y = y

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]