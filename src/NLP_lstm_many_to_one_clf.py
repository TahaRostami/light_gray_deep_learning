"""
This tutorial demonstrates how to train LSTM/BiLSTM/RNN model for binary/multiclass classification using PyTorch.
"""

"""
For simplicity, this tutorial avoids using many third-party libraries, with the exception of pandas.
Pandas is only used for loading the dataset, making the code accessible to a wider audience.
The rest of the code is written purely in Python, or with the use of PyTorch, which is the main framework of interest.
"""

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

"""The line below is optional but commonly used to ensure reproducibility."""
torch.manual_seed(42)

"""
The `load_data` function is responsible for loading the dataset.
In this tutorial, pandas is used exclusively for this purpose.

For simplicity, I generated a sample dataset using ChatGPT, saved in `train.csv` and `validation.csv`.
However, you can replace these with your own dataset. If you do so, the only modification needed 
is to adjust the `load_data` function to correctly load your data.

The tutorial assumes that the dataset is offline, meaning the entire dataset is stored and not streamed. 
This code does not support streaming data, but you can modify it with minimal effort if needed. 
For example, refer to [1] if you need to implement streaming data.

Additionally, the code assumes that the dataset can be loaded into main memory (RAM). 
If your dataset is too large for this, you'll need to make some small adjustments. 
However, if your dataset is offline and fits into RAM, you can simply load it and proceed with the tutorial 
using `X_train`, `y_train`, `X_val`, and `y_val` as shown.
"""
def load_data():
    X_train = pd.read_csv("../data/sentiment/train.csv")
    y_train = X_train['sentiment'].tolist()
    X_train = X_train['content'].tolist()

    X_val = pd.read_csv("../data/sentiment/validation.csv")
    y_val = X_val['sentiment'].tolist()
    X_val = X_val['content'].tolist()

    """
        X_train is a list containing text samples. For example:
        ['The music at the event was incredible and lively', ..., 'The menu was okay but lacked variety']

        Similarly, X_val is a list of validation text samples.

        y_train is a list of corresponding labels for each text sample, such as:
        ['positive', ..., 'neutral']

        y_val is the list of labels for the validation set.
    """
    return X_train, y_train, X_val, y_val



"""
Now we are ready to load our data using the `load_data()` function.
"""
X_train, y_train, X_val, y_val = load_data()


"""
Neural networks require input data to be in numeric form. Since our data consists of text, 
we need to convert it into numerical form through several steps, the first of which is tokenization.
Tokenization is the process of splitting text into smaller chunks, often words or characters.
The most common approach is word-level tokenization, though character-level tokenization is also used.

Below, I implement a simple word-level tokenizer. I'll also demonstrate how you could implement 
a character-level tokenizer. Note that the implementation here is basic for educational purposes.
In practice, more sophisticated tokenizers are typically used. Once you're familiar with PyTorch 
and LSTMs, it will be easy for you to select an appropriate tokenizer from the many available, 
or even implement a custom tokenizer if needed.
"""

def tokenize(X):
    """
        This function returns a list where each item is a list of tokens.
        For example, if X is as follows:
        ['The music at the event was incredible and lively', ...]
        this function will return:
        [['The', 'music', 'at', 'the', 'event', 'was', 'incredible', 'and', 'lively'], ...]
    """
    tokenized_X = [seq.split(' ') for seq in X] # word_level

    """
    Below is an example of a character-level tokenizer.
        # tokenized_X = [[char for char in seq] for seq in X]  # char-level tokenization
    For example, if X is as follows:
    ['The music at the event was incredible and lively', ...]
    this tokenizer will return:
    [['T','h','e',' ','m','u','s','i','c',' ','a','t',' ','t','h','e',' ','e','v','e','n','t',' ','w','a','s',' ',
    'i','n','c','r','e','d','i','b','l','e',' ', 'a','n','d',' ','l','i','v','e','l','y'], ...]
    """
    return tokenized_X

"""
Now, let's tokenize our data.
"""
X_train, X_val = tokenize(X_train), tokenize(X_val)

"""
In addition to tokenization, there are other preprocessing steps in NLP that may help improve model accuracy.
However, this tutorial focuses on PyTorch and LSTM, rather than the broader field of NLP. 
Therefore, we'll skip these additional steps. Once you become comfortable with PyTorch and LSTM, 
learning and adopting these preprocessing techniques will be straightforward.
"""


"""
As mentioned, neural networks require numeric data. Our data consists of two parts: tokens and labels, 
both of which are currently strings. First, I'll write a helper function called `get_lbl2ix` 
that takes all labels as input and returns a dictionary mapping each unique label to a unique numeric index.

For example, if our labels are as follows:
Y = ['positive', 'negative', 'negative', 'neutral', 'positive', 'neutral']
There are three distinct labels:
Unique_Y = ['positive', 'negative', 'neutral']
We can map each unique label to a numeric index, for example:
map_label_to_number = {'positive': 0, 'negative': 1, 'neutral': 2}
"""
def get_lbl2ix(Y):
    lbl2ix = {lbl: i for i, lbl in enumerate(set(Y))}
    return lbl2ix


"""
We also need a helper function that can do the same mapping for tokens.
"""
def get_tok2ix(X):
    """
    In this implementation, I assume that our dataset does not contain the tokens <PAD> and <UNK>
    (shorthand for "unknown"). Neural networks, especially LSTMs, can process sequences of varying lengths,
    but deep learning frameworks like PyTorch require that all sequences in a batch have the same length
    for efficient processing.

    We often pass batches of vectorized text to the neural network.
    However, since different texts within a batch may have different lengths, we need to ensure
    that all texts in a batch are of the same length. This is typically done through padding
    (adding extra tokens to make shorter sequences as long as the longest one) or truncation
    (shortening longer sequences to a specified length).

    Although PyTorch's LSTM can handle inputs of arbitrary lengths, padding is necessary for minibatch processing.
    For example, if you trained an LSTM model on texts with a maximum length of 100, you could still
    pass a text of length 150 during testing or production without any issues. However, if you want to pass
    a batch of texts with varying lengths (e.g., 45, 32, 80, 150, 90), you must ensure they all have
    the same length by padding or truncating them (for example you can use padding to make all of them have size 150).

    To handle this, we usually reserve a special token, <PAD>, that is added to sequences to equalize their lengths.
    You can name this token anything you like, but <PAD> mapped to 0 is a common choice.

    Additionally, we might need another special token, <UNK>, to handle unknown tokens that appear during
    validation or testing but were not present in the training data. If we encounter a token in the validation
    set that was not in the training set, we replace it with <UNK>.

    While there are other special tokens like <START> and <EOF> that could be used, they are not needed
    for this tutorial and are therefore omitted. You can add them by modifying the `tok2ix` dictionary
    initialization as needed.

    The rest of the implementation is similar to `get_lbl2ix`.

    For example, if our X is as follows:
    X = [['it', 'was', 'cool'], ['it', 'is', 'not', 'cool']]
    The distinct tokens are:
    Unique_tokens = ['it', 'was', 'cool', 'is', 'not']
    We map each token to a unique index, resulting in:
    map_token_to_number = {'<PAD>': 0, '<UNK>': 1, 'it': 2, 'was': 3, 'cool': 4, 'is': 5, 'not': 6}
    """
    tok2ix = {'<PAD>': 0, '<UNK>': 1}
    for seq in X:
        for item in seq:
            if item not in tok2ix:
                tok2ix[item] = len(tok2ix)
    return tok2ix

"""
Next, we generate our mappings from tokens to token indices and from labels to label indices.
"""
tok2ix, lbl2ix = get_tok2ix(X_train), get_lbl2ix(y_train)


"""
Using the mappings we've created, we can now convert our text data into numerical format.
Note that the 'UNK' (unknown) token will not appear in the training data because the `tok2ix` mapping 
was created using the training data itself.
"""
# X_train = [[tok2ix[item] if item in tok2ix else tok2ix['<UNK>'] for item in seq] for seq in X_train]
X_train = [[tok2ix[item] for item in seq] for seq in X_train]
y_train = [lbl2ix[lbl] for lbl in y_train]
X_val = [[tok2ix[item] if item in tok2ix else tok2ix['<UNK>'] for item in seq] for seq in X_val]
y_val = [lbl2ix[lbl] for lbl in y_val]

"""
You can run the following commands to verify that everything is working correctly:
print(X_train)
print(X_train[0])
print(y_train)
print(y_train[0])
exit()
"""

"""
In PyTorch, we typically use two main components for preparing data for our models: `Dataset` and `DataLoader`.

While it's not strictly necessary to use the `Dataset` class or `DataLoader`, doing so is common practice. 
It helps organize your code by separating the dataset-related code from the model training code.

There are two types of `Dataset` in PyTorch: Map-style datasets and Iterable-style datasets. 
The latter is useful in scenarios like data streaming, but for most use cases involving offline datasets, 
we use Map-style datasets.

Note: To implement an Iterable-style dataset, you would define a class that inherits from `IterableDataset` 
and customize it by implementing the `__iter__()` method.

Note: To implement a Map-style dataset, you define a class that inherits from `Dataset` and 
customize it by implementing the `__len__()` and `__getitem__()` methods.

Since our data is offline, a Map-style dataset suits our needs. 
Thus, I defined a class named `MyDataset` that inherits from `Dataset`.

While my implementation might not be the most sophisticated, it should provide you with a good understanding 
of how to work with the `Dataset` class.

In the constructor (`__init__`), I accept two arguments: `X` (the vectorized texts) and `y` (the labels). 
The `__len__` function simply returns the number of instances in the dataset. 
Finally, the `__getitem__` method accepts an argument `idx`, which specifies the instance to fetch. 
This method returns three things: the vectorized text, the numerical label, and the length of the vectorized text 
(i.e., the number of tokens in the `idx`-th instance).

You can modify the `__getitem__` method to return more or less information based on your needs. 
Returning `X` and `y` is the minimum required, while including `len(self.X[idx])` can aid in more efficient 
training of the LSTM network. We will later see how `len(self.X[idx])` helps inform the LSTM 
to possibly ignore certain unnecessary parts of the input.

Note: In a more advanced implementation, you might want to load and return your data lazily, 
i.e., only when requested. For simplicity, this tutorial does not cover that, 
but you could try implementing it as a practice exercise.
"""
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], len(self.X[idx])



"""
Now, let's create two instances of the `MyDataset` class: one for the training data 
and one for the validation data.
"""
train_data = MyDataset(X_train, y_train)
val_data = MyDataset(X_val, y_val)

"""
You can verify that everything is working correctly by running the following lines:
print(len(train_data))
print(train_data[0])
exit()
"""

"""
As mentioned earlier, we want to pass a batch of samples to our model at once. PyTorch provides us with 
a utility called `DataLoader`. Given an instance of a `Dataset`, `DataLoader` combines it with a sampler 
and provides an iterable over the dataset.

Note: The `DataLoader` supports both map-style and iterable-style datasets, with options for single- or
multi-process loading, customized loading order, and optional automatic batching (collation) and memory pinning.

You can disable automatic batching and control the process manually using custom collate functions. 
For example, after disabling automatic batching, you can define and pass a custom batching mechanism 
through a `custom_collate_fn` to the `DataLoader`. This can be useful in scenarios where you want 
your batches to have different sizes.

However, in most cases, we rely on the default behavior of the `DataLoader`, i.e., automatic batching 
with a user-specified fixed batch size. Even though automatic batching typically doesn't require 
a `collate_fn`, our specific scenario in this tutorial does.

Every time the `DataLoader` is called, a specified number of samples are extracted from the dataset.
These samples form a 'batch', which is then passed to the `collate_fn` for further processing if needed.

Earlier, we discussed that if we want to pass a batch of data to the LSTM at once, all sequences in the batch 
must have the same size. However, our data contains vectorized text sequences of varying lengths. 
Thus, we need to pad each sequence in a batch to the same size using padding tokens.
"""


def custom_collate_fn(batch):
    """
    This function separates `X`, `y`, and `lengths`.
    Note: Each instance of our dataset consists of three parts: the vectorized text (`X`),
    the numerical label (`y`), and the length of `X`. This structure is based on how we implemented
    the `__getitem__` method in our `MyDataset` class:

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], len(self.X[idx])

    If we had implemented it differently, each instance would be different.
    There is no intrinsic reason for the instances to have this structure other than our implementation.

    When the `DataLoader` is called, it samples a number of instances together, forming a batch.
    We can then unpack the vectorized texts, numerical labels, and the lengths of each vectorized text.

    For example, consider a batch of size three:

    X = [[2,3,1,5],   y = [0,1,0],   lengths = [4,2,3]
         [1,3],
         [1,3,1]]
    """
    X, y, lengths = zip(*batch)

    """Convert lengths to tensors because PyTorch works with tensors"""
    lengths = torch.tensor(lengths, dtype=torch.long)

    """
    Sort by lengths in descending order. While not mandatory, sorting by length in descending order 
    is beneficial when using a technique called `pack_padded_sequence` in PyTorch. 
    This technique can potentially speed up the training process. 
    Sorting ensures that we get the most out of this technique.
    """
    lengths, sorted_idx = lengths.sort(descending=True)
    X = [X[i] for i in sorted_idx]
    y = [y[i] for i in sorted_idx]

    """
    Following our example, after sorting we would have:

    X = [[2,3,1,5],   y = [0,0,1],   lengths = [4,3,2]
         [1,3,1],
         [1,3]]

    However, the sequences still have different lengths. 
    Next, we'll define a function to pad the sequences to make them uniform in length.
    """

    def pad_sequences(sequences, pad_value=0):
        """Find the length of the longest sequence"""
        max_len = max(len(seq) for seq in sequences)

        """Pad each sequence with the `pad_value`"""
        padded_sequences = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

        """Convert the padded sequences to a tensor"""
        return torch.tensor(padded_sequences, dtype=torch.long)

    padded_X = pad_sequences(X, pad_value=0)

    """
    After padding, our example would look like this:

    padded_X = tensor([[2,3,1,5],   y = tensor([0,0,1]),   lengths = tensor([4,3,2])
                       [1,3,1,0],
                       [1,3,0,0]])
    """

    """Convert `y` to a tensor"""
    y = torch.tensor(y, dtype=torch.long)

    """
    Optional Note: Some models may require a masking mechanism, which is a tensor indicating 
    which elements are padding (False) and which are not (True). This can be implemented as:

    mask = padded_X != 0  # True for non-padding elements

    If needed, this is a good place to return mask, and it can be returned from the `collate_fn`:
    return padded_X, y, lengths, mask
    """

    """
    There is no strict requirement on what you return from this function. 
    You can return more or less information based on your needs.
    For our use case, we return `padded_X`, `y`, and `lengths`, which will be passed to our neural network 
    during training or testing.
    """
    return padded_X, y, lengths


"""
Now that we've defined the `custom_collate_fn`, we're ready to create two instances of `DataLoader`:
one for our training data and one for the validation data.

**Key Points:**
1. The `DataLoader` is responsible for fetching batches of data from our dataset and passing them to the model.
2. The `batch_size` parameter defines how many samples are in each batch.
3. The `shuffle` parameter controls whether the data should be shuffled before being passed to the model.

**Note:** 
- It is not mandatory for the `batch_size` of the training and validation data loaders to be the same. 
  However, it is common practice to set them to the same value.
- You can customize the `batch_size` based on your hardware capacity or specific needs.

**Examples:**
1. Different batch sizes:
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
  
   In this example, the training data is loaded in batches of 32, and the validation data in batches of 1.

2. Matching batch sizes:
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
  
   Here, the training data has a batch size of 32, while the validation data has a batch size of 16.

For simplicity and consistency, we'll set both data loaders to use a `batch_size` of 32:
"""
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

"""
To verify that everything is functioning correctly, you can inspect the batches by running the following loop:

#Iterate over the `train_dataloader` and retrieve batches.
for X, Y, lengths in train_dataloader:
    print(X)
    print(Y)
    print(lengths)
    print('.'*100)
exit()

"""


"""
Until now, we've focused on the code related to our data, such as loading and preprocessing. 
Now, we're ready to implement our neural network.

There are different ways to construct a neural network in PyTorch:
1. Creating a hard-coded neural net from scratch.
2. Using `torch.nn.Sequential` (see reference [2] for more details).
3. Defining the network in a class that inherits from `nn.Module`.

In this tutorial, we'll follow approach 3, which is arguably the most important. 
This method is widely used in research and practice, especially for more complex architectures.

**Note:** By inheriting from `nn.Module`, we allow PyTorch to automatically set `Tensor(requires_grad=True)` 
throughout the class, enabling automatic differentiation.
"""

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        Constructor for the LSTM-based neural network model. It accepts four key arguments:

        - vocab_size:    The number of unique tokens in the dataset.
        - embedding_dim: The size of the embedding vectors for each token. This is commonly referred to
                         as word embedding, which represents each token with a vector of size `embedding_dim`.
                         This distributed representation is central to many NLP tasks [3].
        - hidden_dim:    The number of features in the hidden state and cell state of the LSTM,
                         representing the model's capacity to learn long-term dependencies.
        - output_dim:    The number of output classes, corresponding to the unique labels in the dataset.
        """
        """Call the constructor of nn.Module to initialize the model."""
        super(LSTMModel, self).__init__()

        """
        Define the embedding layer:

        - `vocab_size`: The total number of unique tokens.
        - `embedding_dim`: The size of the embedding vector for each token.
        - `padding_idx=0`: Specifies that the embedding for the padding token (index 0) should not be updated during
         training. The embedding layer acts as a lookup table that maps each token index to its corresponding
         embedding vector.

        Although it is possible to create embeddings from scratch, using `nn.Embedding` simplifies the process,
        especially when training the embeddings end-to-end. This means the embedding vectors are updated during 
        training, except for the padding token.
        """
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        """
        Define the LSTM layer:

        - `embedding_dim`: The size of the input features (embedding vectors).
        - `hidden_dim`: The size of the hidden and cell states within the LSTM.
        - `batch_first=True`: Specifies that the input and output tensors are provided with dimensions (batch_size, seq_len, num_features).

        Each input sample in a batch is a sequence of tokens, where each token is represented by a vector of size `embedding_dim`.
        The LSTM processes these sequences and learns to capture dependencies over time steps.

        For example, if `embedding_dim=2` and the input batch `X` is:
        ```
        X = tensor([[2,3,1,5], lengths=tensor([4,3,2])
                    [1,3,1,0]
                    [1,3,0,0]])
        ```
        After passing through the embedding layer, it could look like:
        ```
        X = tensor([[[0.1,0.8],[0.5,0.1],[0.3,0.2],[0.9,0.6]], lengths=tensor([4,3,2])
                    [[0.3,0.2],[0.5,0.1],[0.3,0.2],[0, 0]],
                    [[0.3,0.2],[0.5,0.1],[0, 0],[0, 0]]])
        ```
        The LSTM expects input with shape `(batch_size, seq_len, embedding_dim)`, such as (3, 4, 2) in this case.
        """
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        """
        Define the fully connected layer:

        - `hidden_dim`: The size of the last hidden state from the LSTM.
        - `output_dim`: The number of output classes.

        The LSTM generates a hidden state and cell state for each time step. For classification tasks, we typically use the last hidden state to make predictions. The fully connected layer maps this last hidden state to an output vector of size `output_dim`, which represents the predicted class probabilities.
        """
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        """
        Forward pass of the LSTMModel.

        Parameters:
        - x: Input tensor of shape (batch_size, seq_len), where each value is a token index.
        - lengths: Tensor of shape (batch_size,) indicating the lengths of the sequences in the batch.

        Returns:
        - out: The output of the model, which is the result of applying the fully connected layer to the last hidden state of the LSTM.
        """

        """
        Feed the input tensor through the embedding layer.
        The embedding layer converts each token index into a dense vector of size `embedding_dim`.
        """
        embedded = self.embedding(x)
        """
        Optional: Debugging
        Uncomment the lines below to print the shapes and values for debugging purposes.
        print(embedded)
        print(x.shape)
        print(embedded.shape)
        exit()
        """

        """
        Use the packing the padded sequences technique. This step helps the LSTM process batches more efficiently
        by ignoring padding.
        """
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)

        """
        Optional: Debugging
        Uncomment the lines below to inspect the packed sequence.
        print(packed_embedded)
        exit() 
        """

        """"
        nn.LSTM returns:
         1. packed_output: Packed sequences of LSTM outputs for each time step.
         2. (hidden, cell): The hidden state and cell state at the last time step.
        Make sure all things are alright.
        """
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        """
        Optional: Debugging
        Uncomment the lines below to inspect the outputs from the LSTM.
        print(packed_output)
        print(hidden)
        print(cell)
        exit() 
        """

        """
        Unpack the sequences to restore the original padded format.
        This step is needed if you want to use the sequences for further processing.
        """
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        """
        Optional: Debugging
        Uncomment the lines below to inspect the output after padding.
        print(packed_output)
        print(output)
        exit()
        """

        """
        For classification tasks, there are different approaches one can use.
        For example, there are some common approaches related to our tutorial:
            1- Only use the last hidden state of the LSTM.
               `hidden[-1]` gives us the hidden state of the last time step of the LSTM for each sequence in the batch.
            2- Use max/mean pooling across the time dimension (i.e., dim=1 here)
        In the following we use the first approach. Moreover, I'll show quickly how you can modify the code
        to use max/mean pooling.
        You can see more details about the reasoning behind using these approaches in the Lecture related to LSTM from reference [3].
        """

        out = self.fc(hidden[-1])

        """
        In the case that you want to use max pooling, you need to comment out the above line.
        Then use the following code.
            out=self.fc(torch.max(output, dim=1)[0])
        In the case you want to use mean pooling, after commenting the mentioned line, you
        can use the following code.
            out = self.fc(torch.mean(output, dim=1))
        """

        """
        Note: You can return additional values if needed, but typically for classification,
        you return only the final output from the fully connected layer.
        """
        return out



""" 
Instantiate an instance of the LSTMModel class.
You can adjust `embedding_dim` and `hidden_dim` to suitable values based on your requirements.
"""
model = LSTMModel(len(tok2ix), 50, 100, len(lbl2ix))

"""
Move the model to GPU if a CUDA-compatible device is available.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


"""
Now, we’re ready to start the training and validation phases.

To do this, we first need to specify our loss function and optimizer.

The loss function measures the difference between the model's predicted outputs and the actual target values.
It helps us understand how well or poorly the model is performing. The goal during training is to minimize 
this loss function to improve the model's performance.

We’ll use the Cross-Entropy Loss function, which is commonly applied in classification problems, especially
for multi-class classification tasks.

The optimizer is an algorithm that updates the weights of the neural network to minimize the loss function.
It decides how to adjust the model’s parameters at each training step to reduce the loss.

Here, we'll use the Adam optimizer. It’s an advanced optimizer known for adapting the learning rate for each 
parameter, making it a popular choice due to its efficiency and effectiveness in various scenarios.
"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


"""We define a function for evaluation purposes"""
def calculate_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).sum().item()
    return correct / len(y)


"""
Now we are ready to begin the training and validation phases.
"""
epochs = 10

"""
During training, we iterate over the entire dataset multiple times, with each complete pass referred to as an epoch.
"""
for epoch in range(epochs):

    """
    Set the model to training mode. In this mode, certain layers (e.g., Dropout) behave differently compared to evaluation mode.
    """
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    """
    The DataLoader provides batches of data. For each batch, we extract sequences (X), their corresponding labels (Y), and the lengths of these sequences.
    """
    for X, labels, lengths in train_dataloader:
        X, labels, lengths = X.to(device), labels.to(device), lengths.to(device)

        """
        PyTorch accumulates gradients by default from previous iterations. To avoid this, we clear the previous gradients using optimizer.zero_grad().
        """
        optimizer.zero_grad()

        """Pass the batch (X and lengths) through the model to obtain predictions."""
        outputs = model(X, lengths)

        """Compute the loss by comparing the predicted outputs with the actual labels."""
        loss = criterion(outputs, labels)

        """Calculate gradients of the loss with respect to the model's parameters."""
        loss.backward()

        """Update the model's parameters based on the calculated gradients."""
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += calculate_accuracy(outputs, labels)

    epoch_loss /= len(train_dataloader)
    epoch_accuracy /= len(train_dataloader)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}', end=' ')

    """
    Set the model to evaluation mode. This disables certain layers (e.g., Dropout) that behave differently during training.
    """
    model.eval()
    val_loss = 0
    val_accuracy = 0

    """
    Disable gradient computation during evaluation to save memory and computations.
    """
    with torch.no_grad():
        """
        The DataLoader fetches batches of data for validation. For each batch, we extract sequences (X), their labels (Y), and the lengths of these sequences.
        """
        for X, labels, lengths in val_dataloader:
            X, labels, lengths = X.to(device), labels.to(device), lengths.to(device)

            outputs = model(X, lengths)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels)

    val_loss /= len(val_dataloader)
    val_accuracy /= len(val_dataloader)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}', end='\n')

"""
Optional Note:
In a classification problem using PyTorch's nn.LSTM (or any neural network for that matter), it is common
to use a softmax layer after the nn.Linear layer in the final step of your model's forward method, especially
when you're working with multi-class classification problems.

Why Use Softmax?
Probabilities: The softmax function converts raw scores (logits) from the linear layer into probabilities. 
This is useful for interpreting the output of the model, as the predicted class will have the highest probability.

Multi-Class Classification: In multi-class classification problems, you need to choose one class out of many. 
Softmax makes it easy to determine the class with the highest probability.

Cross-Entropy Loss: When using nn.CrossEntropyLoss, PyTorch automatically applies log-softmax to the model's outputs,
so you don't need to explicitly add a softmax layer if you use this loss function.

When to Add a Softmax Layer
Inference/Deployment: When you're deploying the model and want to get probabilities directly from the model output,
you can apply softmax.
Custom Losses: If you're implementing a custom loss function or need probabilities directly for other calculations,
apply softmax.

When Not to Add a Softmax Layer
Using CrossEntropyLoss: If you're using nn.CrossEntropyLoss, you don't need to add a softmax layer explicitly in the
model. This is because nn.CrossEntropyLoss expects raw logits and internally applies log-softmax before calculating
the loss.
"""


"""
Optional Additional Note:
You can also use Bidirectional LSTM and RNN models.
"""

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Concatenate the final hidden states from both directions
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(hidden_cat)
        return out

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(hidden[-1])
        return out

"""
Related Tutorials:

Reference [3]: This is a free online course offered by Stanford that provides a comprehensive overview of NLP and deep 
learning architectures. It’s excellent for understanding theoretical concepts but does not cover practical implementation
or how to use these models with PyTorch.

Reference [4]: This guide focuses on practical implementation with PyTorch, offering step-by-step instructions for
building and training neural networks. Unlike Reference [3], which is more theoretical, this resource emphasizes 
hands-on experience, covering topics like data preprocessing, model training, and evaluation. It's valuable for 
applying deep learning techniques in real-world scenarios. However, note that this tutorial is somewhat outdated;
for example, it mentions autograd.Variable(requires_grad=True) instead of the more current requires_grad=True with 
tensors, and it uses single samples rather than mini-batches for training. Consequently, some methods described may 
not align with current best practices.

References [5] and [6]: These resources provide clear explanations on preparing data and training LSTM models.
They focus on stock market predictions rather than NLP, which might be confusing if you’re specifically interested 
in NLP, especially with the use of Embedding Layers in this tutorial.

Reference [7]: This reference teaches how to train an RNN model from scratch. While it offers a foundational 
understanding of building RNNs, the code provided is somewhat ad hoc, which might make it challenging for beginners 
to grasp and apply broadly.

Reference [8]: This is a highly recommended hands-on guide for preparing data and training LSTM models. However, it 
addresses a many-to-many task, whereas this tutorial focuses on a many-to-one task. Thus, while the methodologies are 
kind of similar, the specific application differs.

"""

"""
References: 

[1] - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
[2] - https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
[3] - https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4
[4] - https://github.com/rguthrie3/DeepLearningForNLPInPytorch/blob/master/Deep%20Learning%20for%20Natural%20Language%20Processing%20with%20Pytorch.ipynb
[5] - https://www.youtube.com/watch?v=CZi5Avp6p1s&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi&index=47&t=89s&pp=iAQB
[6] - https://www.youtube.com/watch?v=hIQLy5zCgH4&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi&index=48&pp=iAQB
[7] - https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
[8] - https://www.kdnuggets.com/2018/06/taming-lstms-variable-sized-mini-batches-pytorch.html
"""