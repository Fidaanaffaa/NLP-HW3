import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt
from plotnine import *
import os
from statistics import mean


# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
NEGAT_POLARITY ="NEGATE_POLARITY"
RARE = "RARE"
LOG_LINEAR_FILE_PATH = "/log_linear_model5.pkl"
W2V_LOG_LINEAR_FILE_PATH = "/log_linear_w2v_model5.pkl"
W2V_LSTM_FILE_PATH = "/LSTM_w2v_model5.pkl"

# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    # vocab = list(wv_from_bin.vocab.keys())
    # print(wv_from_bin.vocab[vocab[0]])
    print(wv_from_bin.key_to_index[vocab[0]])

    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    # embedding_sent_mean = np.mean(sentence_to_embedding(sent, word_to_vec, 1, embedding_dim), axis=0)
    embedding_sent_mean = np.mean(sentence_to_embedding(sent, word_to_vec, 1, embedding_dim), axis=0)

    return embedding_sent_mean


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """

    zero_peding_vec = np.zeros(size)
    zero_peding_vec[ind] = 1
    one_hot_vec = zero_peding_vec
    return one_hot_vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vocabulary = word_to_ind.keys()
    text_list = sent.text
    # test if all words in the sentence are unknown words
    i = 0
    for word in text_list:
        if i == len(text_list):
            return np.zeros(len(text_list))
        if word in vocabulary:
            break
        else:
            i += 1
    # initialize param
    i = 0
    one_hot_vectors_matrix = np.zeros(shape=((len(text_list), len(word_to_ind))))
    for word in text_list:
        one_hot_vectors_matrix[i] = get_one_hot(len(word_to_ind), word_to_ind[word])
        i += 1
    sum_vectors = one_hot_vectors_matrix.sum(axis=0)  # axis = 0 sum all the cols
    embedding_vector = sum_vectors / len(word_to_ind)  # normalize the sum vector

    return embedding_vector


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word2vec_dict, vocabulary = {}, set()

    word2vec_curr_index = 0
    for word in words_list:
        if word not in word2vec_dict.keys():
            vocabulary.add(word)
            word2vec_dict[word] = word2vec_curr_index
            word2vec_curr_index += 1
    return word2vec_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """

    vectors_embedding_matrix = np.zeros(shape=(seq_len, embedding_dim))
    n = len(sent.text) if len(sent.text) < seq_len else seq_len
    for i in range(n):
        cur_word = sent.text[i]
        if cur_word in word_to_vec.keys():
            vectors_embedding_matrix[i] = word_to_vec[cur_word]
    return vectors_embedding_matrix



class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # using the 2 help function to add subset for negate polartiy and rare for easier future uses
        self.add_negate_polarity()
        self.add_rare()
        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}


    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


    def add_negate_polarity(self):
        # DataManager obj uses -> sentimentTreeBack obj uses -> Sentence obj
        negate_polarity_sentences = [self.sentiment_dataset.sentences[i] for i in data_loader.get_negated_polarity_examples(self.sentiment_dataset.sentences)]
        self.sentences[NEGAT_POLARITY] = negate_polarity_sentences
    def add_rare(self):
        rare_sentences = [self.sentiment_dataset.sentences[i] for i in data_loader.get_rare_words_examples(self.sentiment_dataset.sentences, self.sentiment_dataset)]
        self.sentences[RARE] = rare_sentences

# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    # implementing bi LSTM which is 2 direction RNN meaning
    # from start to end and then use activation layer from end to start
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self._bi_LSTM_model = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        # batch_first=True meaning that we must have the batch as the first dim
        # meaning input x need to have the shape ->[ batch_size ,sequence length , input/feature size]
        self._num_layers = n_layers
        self._hidden_size = hidden_dim
        self._dropout = nn.Dropout(p=dropout)
        # we implement bi LSTM so need to give for each input the new leayer and the layer before
        self._fully_connected_linear_layer = nn.Linear(in_features=hidden_dim*2, out_features=1)
        # we do hidden_dim*2 since we have 1 layer go forward and 1 go backward but they all get concatenated
        # for the same hidden state
        self._activation = nn.Sigmoid()
        return

    def forward(self, text):
        # hidden0 = torch.zeros((self._num_layers*2, text.size(0), self._hidden_size))
        # cell_state0 = torch.zeros((self._num_layers*2, text.size(0), self._hidden_size))
        # out_before_drop, _ = self._bi_LSTM_model(text, (hidden0, cell_state0)) # _ == (hidden_state, cell_state)
        # out_before_drop = out_before_drop[:, -1, :] # we want to decode the hiden state only of the last state
        # # so we will take all the sample in our batch, but only in the last time steps
        # # so from out (batch_size ,sequence length , input/feature size) -> (batch_size, input/feature size)
        # drop = self._dropout(out_before_drop)
        # out = self._fully_connected_linear_layer(drop)
        # return out
        h0 = torch.zeros(2*self._num_layers, text.size(0), self._hidden_size)
        c0 = torch.zeros(2*self._num_layers, text.size(0), self._hidden_size)

        output, (hn, cn) = self._bi_LSTM_model(text, (h0, c0))
        cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        dropped_output = self._dropout(cat)
        final = self._fully_connected_linear_layer(dropped_output)
        return final

    def predict(self, text):
        # hidden0 = torch.zeros((self._num_layers*2, text.size(0), self._hidden_size))
        # cell_state0 = torch.zeros((self._num_layers*2, text.size(0), self._hidden_size))
        # out_before_drop, _ = self._bi_LSTM_model(text, (hidden0, cell_state0))
        # out_before_drop = out_before_drop[:, -1, :]
        # drop = self._dropout(out_before_drop)
        # out = self._activation(self._fully_connected_linear_layer(drop))
        # return out
        return self._activation(self.forward(text))



class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self._log_linear_layer = nn.Linear(in_features=embedding_dim, out_features=1)  # the input should a vector of len embedding_dim
        # self._activation = nn.Sigmoid()
        # self._activation_func = nn.Tanh()
        return

    def forward(self, x):
        # remark for me - the forward func is like the "output" the model give us meaning the model prediction
        # because we want to use a soft max, wer will use the model prediction out put
        # and our prediction will be in the predict method after the using the activation func
        # the sigmoid func in our case. in our case its like the inverse function for the predict func!
        # remark for the future - in pytorch if will use nn.BCELoss() as a criterion in the last layer
        #  the loss func already use a softmax fun so we should not us any softmax fun at the end
        # so ill use the nn.BCEWithLogitsLoss() who dont instead
        return self._log_linear_layer(x)

    def predict(self, x):
        # return self._activation_func(self._log_linear_layer(x))
        layer_o = self._log_linear_layer(x)
        out = nn.Sigmoid()(layer_o)
        return out


# ------------------------- training functions -------------

#TODO 1 (section 5 in ex3)
def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    preduction_round = [0 if  cur_pred < 0.5 else  1 for cur_pred in preds]
    match_prediction = 0
    for i in range(len(preds)):
        if preduction_round[i] == y[i]:
            match_prediction +=1
    return match_prediction / len(preds)

def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    #TODO i think i need to change everyting to torch.tensor arr so the updateing gradient will be saved to use
    # forward and all this shit
    ## set our model to train method
    model.train()
    loss_func = criterion
    activation_func = nn.Sigmoid()
    accuracy_list, loss_list = [],  []
    j = 1
    for word_batch in data_iterator:
        # print(f'train_epoch number = {j}')
        word_sample, tag_sample = word_batch[0].float(), word_batch[1].float()

        # step 1 forward pass - predict and compute loss
        # empty the gradient cuz every time we call backward it will
        # write our gradient and accumulate it in w.grad attribute
        optimizer.zero_grad()
        y_predict = model.forward(word_sample)
        # y_predict = model._activation(y_predict)
        # y_predict = nn.Sigmoid()(y_predict)


        y_real = tag_sample.view(tag_sample.shape[0], 1)
        # y_real = word_batch[1].view(tag_sample.shape[0], 1)
        loss_tensor = criterion(y_predict, y_real)

        # loss_tensor = loss_func(y_predict, y_real)
        loss_list.append(loss_tensor.item())



        # step 2 - pytorch compute and update the local gradient so no need
        # to implement step 2 meaning the local gradient computation

        # step 3 - backward pass pytorch also compute the backward computation for us
        loss_tensor.backward()

        # update weight
        optimizer.step() # optimize and update our gradient weight

        # y_predict_with_sigmoid = torch.sigmoid(y_predict)
        y_predict_with_sigmoid =  activation_func(y_predict)

        curr_accuracy = binary_accuracy(y_predict_with_sigmoid, y_real)
        accuracy_list.append(curr_accuracy)
        j += 1
    mle_result = mean(loss_list)
    average_accuracy = mean(accuracy_list)
    return mle_result, average_accuracy


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    # set our model to evaluation method
    model.eval()
    loss_func = criterion
    activation_func = nn.Sigmoid()
    accuracy_list, loss_list = [],  []
    k = 1
    for word_batch in data_iterator:
        # print(f'train evaluate  number = {k}')

        word_sample, tag_sample = word_batch[0].float(), word_batch[1].float()
        y_predict = model.forward(word_sample)
        # y_predict = model.predict()

        # y_predict = model.predict(word_batch.view(tuple(word_sample.size())).float())

        # y_real = word_batch[1].view(tag_sample.shape[0], 1)
        y_real = word_batch[1].view(tag_sample.shape[0], 1)
        # print(f"y_ perdict is : {y_predict},  y_real is: {y_real}\n  word_sample is : {word_sample}, tag_sample is : {tag_sample}")
        # curr_accuracy = binary_accuracy(y_predict, y_real)
        # accuracy_list.append(curr_accuracy)
        # loss_tensor = loss_func(y_predict, y_real)
        loss_tensor = criterion(y_predict, y_real)

        loss_list.append(loss_tensor.item())


        # y_predict_with_sigmoid = torch.sigmoid(y_predict)
        y_predict_with_sigmoid =  activation_func(y_predict)
        curr_accuracy = binary_accuracy(y_predict_with_sigmoid, y_real)

        accuracy_list.append(curr_accuracy)

        k += 1
    # print(f"loos validation list: {loss_list} ,\n  validation acc list: {accuracy_list}")
    mle_result = mean(loss_list)
    average_accuracy = mean(accuracy_list)
    # print(f"mean loos validation list: {mle_result} ,\n  mean validation acc list: {average_accuracy}")

    return mle_result, average_accuracy



def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    prediction_list = [model.predict(word_batch[0]) for word_batch in data_iter]
    return torch.tensor(prediction_list)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_loss_list, train_accuracy_list = np.zeros(n_epochs), np.zeros(n_epochs)
    validation_loss_list, validation_accuracy_list = np.zeros(n_epochs), np.zeros(n_epochs)

    solver = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.BCEWithLogitsLoss()
    train_data_iterator = data_manager.get_torch_iterator(data_subset=TRAIN)
    validation_data_iterator = data_manager.get_torch_iterator(data_subset=VAL)
    # training loop
    # for i in range(20):

    for i in range(n_epochs):
        print("epuch num cur i = " , i)
        train_loss_list[i], train_accuracy_list[i] = train_epoch(model, train_data_iterator, solver, nn.BCEWithLogitsLoss())
        validation_loss_list[i], validation_accuracy_list[i] = evaluate(model, validation_data_iterator, nn.BCEWithLogitsLoss())

    return train_loss_list, train_accuracy_list, validation_loss_list, validation_accuracy_list, solver


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # meta data
    # learning_rate, n_epoch, batches_size = 0.01, 20, 64
    w_decay = 0.001
    learning_rate, n_epoch, batches_size = 0.01, 20, 64

    # n_epoch = 5
    # batches_size = 20

    # models and objects data
    train_data_manger = DataManager(data_type=ONEHOT_AVERAGE, batch_size=batches_size)
    test_sub_set = train_data_manger.get_torch_iterator(data_subset=TEST)
    embedding_dim = train_data_manger.get_input_shape()[0]  # need to unpack it cuz its a tuple
    log_linear_model = LogLinear(embedding_dim=embedding_dim)
    loss_func = nn.BCEWithLogitsLoss()
    model_file_path = LOG_LINEAR_FILE_PATH
    # model_file_path = "/log_linear_model.pkl"
    model_result_path = "/log_linear_result.txt"
    # training phase
    train_loss, train_accuracy, train_validation_loss, train_validation_accuracy, trained_solver = train_model(log_linear_model,
                                                                                                       train_data_manger,
                                                                                                       n_epoch,
                                                                                                       learning_rate,
                                                                                                       w_decay)
    # saving the model check point for models comparison usage
    save_model(log_linear_model, os.getcwd() + model_file_path,  n_epoch, trained_solver)


    print(f'train loss list is = {train_loss} \n train accuracy list is = { train_accuracy}')
    print(f'validation loss list is = {train_validation_loss} \n validation accuracy list is = { train_validation_accuracy}')

    plot_result_per_epoch_iter(train_loss, train_validation_loss, n_epoch, w_decay, "Log Linear One Hot Vec LOSS" ,"Train Loss ", "Validation Loss")
    plot_result_per_epoch_iter(train_accuracy, train_validation_accuracy, n_epoch, w_decay, "Log Linear One Hot Vec Accuracy ", "Train Accuracy", "Validation Accuracy")

    # evaluate phase
    test_loss, test_accuracy = evaluate(log_linear_model, test_sub_set, loss_func)
    with open(os.getcwd() + model_result_path, "w") as model_result:
        model_result.write(f"Log Linear Model Test Loss is: {test_loss} \n")
        model_result.write(f"Log Linear Model Test Validation is: {test_accuracy} \n")

    return
    # return train_loss, train_accuracy, train_validation_loss, train_validation_accuracy

def plot_result_per_epoch_iter(loss_list, accuracy_list, n_epoch, weight_decay_val,title: str,  loss_legend : str, acc_legend: str):


    #
    domain = np.arange(1, n_epoch + 1)
    plt.plot(domain, np.array(loss_list), color="purple")
    plt.plot(domain, np.array(accuracy_list), color="red")
    plt.title( f"{title}  \n for different number of epoch iteration with w = {weight_decay_val}")
    plt.legend([ loss_legend, acc_legend], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()



def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """

    # meta data
    # learning_rate, n_epoch, batches_size = 0.01, 20, 64
    w_decay = 0.001
    learning_rate, n_epoch, batches_size = 0.01, 20, 64

    # models and objects data
    train_data_manger = DataManager(data_type=W2V_AVERAGE,embedding_dim=W2V_EMBEDDING_DIM,batch_size=batches_size)
    test_sub_set = train_data_manger.get_torch_iterator(data_subset=TEST)
    embedding_dim = train_data_manger.get_input_shape()[0]  # need to unpack it cuz its a tuple
    log_linear_model = LogLinear(embedding_dim=embedding_dim)
    loss_func = nn.BCEWithLogitsLoss()
    # model_file_path = "/log_linear_w2v_model.pkl"
    model_file_path = W2V_LOG_LINEAR_FILE_PATH

    model_result_path = "/log_linear_w2v_result.txt"
    # training phase
    train_loss, train_accuracy, train_validation_loss, train_validation_accuracy, trained_solver = train_model(log_linear_model,
                                                                                                       train_data_manger,
                                                                                                       n_epoch,
                                                                                                       learning_rate,
                                                                                                       w_decay)
    # saving the model check point for models comparison usage
    save_model(log_linear_model, os.getcwd() + model_file_path,  n_epoch, trained_solver)


    print(f'train loss list is = {train_loss} \n train accuracy list is = { train_accuracy}')
    print(f'validation loss list is = {train_validation_loss} \n validation accuracy list is = { train_validation_accuracy}')

    plot_result_per_epoch_iter(train_loss, train_validation_loss, n_epoch, w_decay, "Log Linear W2V LOSS " ,"Train Loss ", "Validation Loss")
    plot_result_per_epoch_iter(train_accuracy, train_validation_accuracy, n_epoch, w_decay, "Log Linear W2V Accuracy ", "Train Accuracy", "Validation Accuracy")


    return




def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    # meta data
    # learning_rate, n_epoch, batches_size = 0.01, 20, 64
    w_decay, drop_out , hidden_dim = 0.0001, 0.5, 100
    learning_rate, n_epoch, batches_size = 0.001, 4, 64

    # models and objects data
    train_data_manger = DataManager(data_type=W2V_SEQUENCE,embedding_dim=W2V_EMBEDDING_DIM,batch_size=batches_size)
    test_sub_set = train_data_manger.get_torch_iterator(data_subset=TEST)
    embedding_dim = train_data_manger.get_input_shape()[1]  # for LTS the din is in shape 1

    lstm_model = LSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=1, dropout=drop_out)
    loss_func = nn.BCEWithLogitsLoss()
    # model_file_path = "/LSTM_w2v_model.pkl"
    model_file_path = W2V_LSTM_FILE_PATH

    model_result_path = "/LSTM_w2v_result.txt"
    # training phase
    train_loss, train_accuracy, train_validation_loss, train_validation_accuracy, trained_solver = train_model(lstm_model,
                                                                                                       train_data_manger,
                                                                                                       n_epoch,
                                                                                                       learning_rate,
                                                                                                       w_decay)
    # saving the model check point for models comparison usage
    save_model(lstm_model, os.getcwd() + model_file_path,  n_epoch, trained_solver)


    plot_result_per_epoch_iter(train_loss, train_validation_loss, n_epoch, w_decay, "LOSS LSTM" ,"Train Loss ", "Validation Loss")
    plot_result_per_epoch_iter(train_accuracy, train_validation_accuracy, n_epoch, w_decay, "Accuracy LSTM ", "Train Accuracy", "Validation Accuracy")
    return

def subset_result_on_trained_models(log_linear_data_manager, w2v_log_linear_data_manager, w2v_lstm_data_manager):
    """
    load the trained model from a pkl file and calculate the subset result
    :param log_linear: log linear data manger obj
    :param w2v_log_linear: w2v log linear  data manger obj
    :param w2v_lstm: w2v lstm model  data manger obj
    :return:
    """
    # log linear meta data
    log_linear_w_decay = 0.001
    learning_rate = 0.01

    # LSTM meta data
    lstm_w_decay, drop_out, hidden_dim = 0.0001, 0.5, 100
    lstm_learning_rate = 0.001

    # initialize  models, optimizers, and lost func
    log_linear = LogLinear(embedding_dim=log_linear_data_manager.get_input_shape()[0])
    w2v_log_linear = LogLinear(embedding_dim=w2v_log_linear_data_manager.get_input_shape()[0])
    w2v_lstm = LSTM(embedding_dim=w2v_lstm_data_manager.get_input_shape()[1], hidden_dim=hidden_dim, n_layers=1, dropout=drop_out)
    log_linear_optimizer = torch.optim.Adam(params=log_linear.parameters(), lr=learning_rate, weight_decay=log_linear_w_decay)
    w2v_log_linear_optimizer = torch.optim.Adam(params=w2v_log_linear.parameters(), lr=learning_rate, weight_decay=log_linear_w_decay)
    w2v_lstm_optimizer = torch.optim.Adam(params=w2v_lstm.parameters(), lr=lstm_learning_rate, weight_decay=lstm_w_decay)
    loss_func = nn.BCEWithLogitsLoss()
    # model, optimizer, epoch
    trained_log_linear, _, log_linear_epoch= load(log_linear, os.getcwd() + LOG_LINEAR_FILE_PATH, log_linear_optimizer)
    trained_w2v_log_linear, _, w2v_log_linear_epoch = load(w2v_log_linear, os.getcwd() + W2V_LOG_LINEAR_FILE_PATH, w2v_log_linear_optimizer)
    trained_w2v_lstm, _, w2v_lstm_epoch = load(w2v_lstm, os.getcwd() + W2V_LSTM_FILE_PATH, w2v_lstm_optimizer)

    # evaluate on different subset data

    log_linear_lost, log_linear_acc = evaluate(trained_log_linear, log_linear_data_manager.get_torch_iterator(data_subset=NEGAT_POLARITY), loss_func)
    print(f"One hot vector log linear model evaluation with the trained model \n NEGAT_POLARITY LOSS:   {log_linear_epoch}, NEGAT_POLARITY ACC: {log_linear_acc}")
    log_linear_lost, log_linear_acc = evaluate(trained_log_linear, log_linear_data_manager.get_torch_iterator(data_subset=RARE), loss_func)
    print(f"One hot vector log linear model evaluation with the trained model \n RARE LOSS:   {log_linear_epoch}, REARE ACC: {log_linear_acc}")

    w2v_log_linear_lost, w2v_log_linear_acc = evaluate(trained_w2v_log_linear, w2v_log_linear_data_manager.get_torch_iterator(data_subset=NEGAT_POLARITY), loss_func)
    print(f"W2V log linear model evaluation with the trained model \n NEGAT_POLARITY LOSS:   {w2v_log_linear_lost}, NEGAT_POLARITY ACC: {w2v_log_linear_acc}")
    w2v_log_linear_lost, w2v_log_linear_acc = evaluate(trained_w2v_log_linear, w2v_log_linear_data_manager.get_torch_iterator(data_subset=RARE), loss_func)
    print(f"W2V log linear model evaluation with the trained model \n RARE LOSS:   {w2v_log_linear_lost}, RARE ACC: {w2v_log_linear_acc}")

    w2v_lstm_lost, w2v_lstm_acc = evaluate(trained_w2v_lstm, w2v_lstm_data_manager.get_torch_iterator(data_subset=NEGAT_POLARITY), loss_func)
    print(f"W2V log linear model evaluation with the trained model \n NEGAT_POLARITY LOSS:   {w2v_lstm_lost}, NEGAT_POLARITY ACC: {w2v_lstm_acc}")
    w2v_lstm_lost, w2v_lstm_acc = evaluate(trained_w2v_lstm, w2v_lstm_data_manager.get_torch_iterator(data_subset=RARE), loss_func)
    print(f"W2V log linear model evaluation with the trained model \n RARE LOSS:   {w2v_lstm_lost}, RARE ACC: {w2v_lstm_acc}")



# def text():
#     train_loss_list, train_accuracy_list = np.zeros(3), np.zeros(3)
#     print(train_loss_list, "\n", train_accuracy_list)
#     return


if __name__ == '__main__':
#     # examples for reading the sentiment dataset
#     dataset = data_loader.SentimentTreeBank()
#     # get train set
#     print(dataset.get_train_set()[:2])
#     print(dataset.get_train_set()[0].sentiment_val)
#     # get word counts dictionary
#     print("3")
#     print(list(dataset.get_word_counts().keys())[:10])
#

    train_data_manger = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)
    test_set = train_data_manger.get_torch_iterator(data_subset=TEST)
    plt.clf()
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
    #
    log_linear_data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)
    w2v_log_linear_data_manager = DataManager(data_type=W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=64)
    w2v_lstm_data_manager = DataManager(data_type=W2V_SEQUENCE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=64)
    subset_result_on_trained_models(log_linear_data_manager, w2v_log_linear_data_manager, w2v_lstm_data_manager)