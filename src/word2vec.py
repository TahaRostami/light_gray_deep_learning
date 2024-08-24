"""
This code demonstrates how to train a basic word2vec model using PyTorch. By "basic,"
I mean that it does not incorporate advanced techniques such as negative sampling.
This example is provided as a foundational example rather than a comprehensive
tutorial, given the abundance of existing resources on word2vec. It is included here
for completeness and as a reference point. A detailed tutorial will follow in the
next part, where we will explore working with LSTM networks.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


"""Set a manual seed for reproducibility of results."""
torch.manual_seed(42)

"""
Specify the device for computation: GPU if available, otherwise CPU.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


"""
Define the `load_data` function to load and return the dataset as a single string.
"""
def load_data():
        text = (
            """Once upon a time in a dense forest there lived a clever fox The fox was known throughout the forest
               for its cunning and intelligence One day as the fox was wandering near the edge of the forest it
               spotted a crow perched on a tree branch holding a piece of cheese in its beak
               The clever fox thought to itself If I could only get that piece of cheese I would have a feast
               So the fox approached the crow and began to praise it Oh what a beautiful bird you are the fox 
               exclaimed Your feathers are so shiny and your voice must be just as lovely Could you sing a song
               for me The crow flattered by the fox words opened its beak to sing As it did the piece of cheese
               fell to the ground The fox quickly grabbed the cheese and ran away leaving the crow to regret its
               foolishness"""
        )
        return text.lower()


data = load_data()


"""
Convert the text data into a list of tokens (words) using simple whitespace splitting.
"""
def tokenize(data):
    return data.split()

data=tokenize(data)


"""
Create a mapping from tokens to unique indices. This mapping is essential for converting text data into numerical data.
"""
def get_tok2ix(tokenized_data):
    tok2ix = {'<UNK>': 0}
    for tok in tokenized_data:
        if tok not in tok2ix:
            tok2ix[tok] = len(tok2ix)
    return tok2ix


tok2ix = get_tok2ix(data)


"""
Create a reverse mapping from indices to tokens for decoding the indices back into readable tokens.
"""
def get_ix2tok(tok2ix):
    return {ix: tok for tok, ix in tok2ix.items()}


ix2tok = get_ix2tok(tok2ix)

"""
Prepare the data for training by converting tokens into their corresponding indices and generating context-target pairs.
"""
def prepare_data(tokenized_data, tok2ix=None, context_size=3):
    X, Y = [], []
    for i in range(len(tokenized_data) - context_size):
        context = tokenized_data[i:i + context_size]
        target = tokenized_data[i + context_size]

        if tok2ix is not None:
            context = [tok2ix.get(tok, tok2ix['<UNK>']) for tok in context]
            target = tok2ix.get(target, tok2ix['<UNK>'])

        X.append(context)
        Y.append(target)

    return X, Y


X, Y = prepare_data(data, tok2ix=tok2ix, context_size=10)


"""
Define a custom Dataset class to handle the input and target pairs for training. This allows PyTorch's DataLoader
to handle batching and shuffling efficiently.
"""
class TranslationDataset(Dataset):
    def __init__(self, X,Y):
        self.X,self.Y = X,Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor([self.Y[idx]], dtype=torch.long)



train_dataset = TranslationDataset(X,Y)
"""
Create a DataLoader to iterate over the dataset, fetching samples in batches and optionally shuffling them.
"""
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)


"""
Define the Word2Vec model. This version supports batch processing and uses a simple feed-forward network.
"""
class Word2VecModel(nn.Module):
    """Adapted from Reference [1]. Modified to support batch inputs instead of individual inputs."""

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        """
        Reshape the embeddings to (batch_size, context_size * embedding_dim) for batch processing.
        """
        embeds = self.embeddings(inputs).view(inputs.shape[0], -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

model = Word2VecModel(len(tok2ix), 64, len(X[0])).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

"""
Train the model for a specified number of epochs. For each epoch, iterate over the data, compute the loss,
perform backpropagation, and update the model parameters.
"""
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for context, target in train_dataloader:
        context = context.to(device)
        target = target.to(device)
        model.zero_grad()
        output = model(context)
        loss = criterion(output, target.squeeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    loss = epoch_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs} Loss: {loss:.4f}')


"""
Define a subset of tokens to visualize in the 2D embedding space.
"""
subset_tokens = ['fox','cunning','tree','forest']# list(tok2ix.keys())

"""
Plot the word embeddings for the subset of tokens. Use t-SNE to reduce the dimensionality to 2D for visualization.
Note: The t-SNE perplexity parameter is set to the number of unique tokens minus one to ensure proper functioning.
"""
def plot_embeddings(model, ix2tok, subset_tokens):
    embeddings = model.embeddings.weight.data.numpy()
    dim_reducer = TSNE(n_components=2, random_state=42, perplexity=len(ix2tok)-1)
    dim_reduced = dim_reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(dim_reduced):
        if ix2tok[i] in subset_tokens:
            plt.scatter(x, y)
            plt.text(x, y, ix2tok[i], fontsize=9)
    plt.title('Reduced to 2D Word Embeddings (Subset)')
    plt.grid(True)
    plt.show()


plot_embeddings(model, ix2tok, subset_tokens)

"""
References: 

[1] - https://github.com/rguthrie3/DeepLearningForNLPInPytorch/blob/master/Deep%20Learning%20for%20Natural%20Language%20Processing%20with%20Pytorch.ipynb
"""

