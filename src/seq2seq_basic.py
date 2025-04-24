"""
This tutorial demonstrates how to train a basic seq2seq model for machine translation.
By "basic," we mean that this tutorial uses a simple greedy approach for decoding rather
than more advanced techniques like beam search. Additionally, the model described here
does not incorporate an attention mechanism.
"""

import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



"""
We set a manual seed for reproducibility. This ensures that the results are consistent
each time the code is run, which is especially important during debugging or comparison.
"""
torch.manual_seed(42)


"""
We specify the device to be used for computation: either GPU (if available) or CPU.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"


"""
We define special tokens that will be used throughout the model.
We assumed the training and evaluation data do not contain these tokens inherently.
"""

"""<PAD>: Padding token used to ensure that all sequences in a batch have the same length."""
PAD_token = "<PAD>"  # 0

"""
<SOS>: Start of Sequence token used to indicate the beginning of a sequence.
It helps the model recognize when to start processing or generating a sequence.
"""
SOS_token = "<SOS>"  # 1

"""
<EOS>: End of Sequence token used to signal the end of a sequence.
The model can also output this token to indicate that it has finished its prediction.
"""
EOS_token = "<EOS>"  # 2

"""
<UNK>: Unknown token used for words or tokens in the validation/test set that were not seen during training.
This allows the model to handle out-of-vocabulary (OOV) words more gracefully.
"""
UNK_token = "<UNK>"  # 3


"""
We load the dataset, which consists of English to French sentence pairs.
The data is formatted as:
    [[English sentence 1, French translation 1], ..., [English sentence n, French translation n]]
We also remove certain punctuation marks during this process. 
The load_data function accepts an optional parameter `n` which specifies the number of items to load. 
If `n` is None, the entire dataset is loaded.
"""
def load_data(n=None):
    pairs = []
    """I downloaded the dataset from https://download.pytorch.org/tutorial/data.zip"""
    with open("../data/eng-fra.txt", 'r', encoding='latin-1') as f:
        for line in f.readlines():
            if n is not None and n==len(pairs):
                break
            pairs.append(
                line.replace('\n', '').
                replace('.', '').replace('!', '').
                replace('?', '').lower().split('\t'))
    return pairs


"""
For faster training during this tutorial, we load a subset of the data.
In a real-world scenario, you would likely load the entire dataset.
"""
pairs = load_data(n=15000)


"""
Next, we will tokenize the sequences. We will write a function called `tokenize` 
that prepares the sentences for model training by converting them into lists of tokens.
"""
def tokenize(pairs):
    tokenized = []
    for en_sentence, fr_sentence in pairs:
        """
                Here, we use a simple tokenizer that splits text based on spaces (' ').
                However, more advanced tokenizers can be employed. For example:

                    import spacy
                    nlp_en = spacy.load("en_core_web_sm")
                    nlp_fr = spacy.load("fr_core_news_sm")
                    tokenized_en = [SOS_token] + [token.text for token in nlp_en(en_sentence)] + [EOS_token]
                    tokenized_fr = [SOS_token] + [token.text for token in nlp_fr(fr_sentence)] + [EOS_token]

                Regardless of the tokenizer used, we add the `SOS_token` and `EOS_token` 
                to the beginning and end of each sentence to mark the sequence boundaries.

                Note: Adding these tokens is common practice for seq2seq models, but other 
                approaches might not include them, depending on the model architecture and application.
        """
        tokenized_en = [SOS_token] + en_sentence.split(' ') + [EOS_token]
        tokenized_fr = [SOS_token] + fr_sentence.split(' ') + [EOS_token]
        tokenized.append([tokenized_en, tokenized_fr])
    return tokenized


pairs = tokenize(pairs)

"""
We will use the majority of the data for training, but we'll set aside a small portion for testing.
From this point onward, the `pairs` variable will contain our training data, 
and `test_pairs` will contain our test data.
"""
random.seed(42)
random.shuffle(pairs)

test_pairs = pairs[:15]
pairs = pairs[len(test_pairs) + 1:]

"""
We assign a unique index to each token in both the English and French vocabularies.
This allows us to train separate embedding layers for the source (English) and target (French) languages.
However, note that this is just one approach. Alternatively, a single embedding layer could be used 
to learn embeddings for the combined vocabulary of both languages.
"""
def get_tok2ix(tokenized_pairs):
    tok2ix = {'src_lang': {PAD_token: 0, SOS_token: 1, EOS_token: 2, UNK_token: 3},
              'trg_lang': {PAD_token: 0, SOS_token: 1, EOS_token: 2, UNK_token: 3}}
    for pair in tokenized_pairs:
        for i, p in enumerate(pair):
            lang = 'src_lang' if i == 0 else 'trg_lang'
            for tok in p:
                if tok not in tok2ix[lang]:
                    tok2ix[lang][tok] = len(tok2ix[lang])
    return tok2ix


"""
We pass the tokenized training data to the get_tok2ix function to create a dictionary 
that maps each unique token to an index.
"""
tok2ix = get_tok2ix(pairs)

"""
Later, our model will output sequences of indices corresponding to the vocabulary tokens.
To interpret these indices as human-readable text, we need to reverse the mapping 
from indices back to tokens. The get_ix2tok function creates this reverse mapping.
"""
def get_ix2tok(tok2ix):
    ix2tok = {'src_lang': {}, 'trg_lang': {}}
    for lang in tok2ix:
        for k in tok2ix[lang]:
            ix2tok[lang][tok2ix[lang][k]] = k
    return ix2tok


"""
We create the reverse mapping (ix2tok) by passing our tok2ix dictionary 
to the get_ix2tok function.
"""
ix2tok = get_ix2tok(tok2ix)



"""
Since PyTorch models operate on numerical data, we need to convert (or vectorize) 
our tokenized sequences into sequences of corresponding indices.
"""
def vectorize_tokenized_pairs(tokenized_pairs, tok2ix):
    vectorized_pairs = []
    for pair in tokenized_pairs:
        vectorized_pair = []
        for i, p in enumerate(pair):
            lang = 'src_lang' if i == 0 else 'trg_lang'
            vectorized_sentence = []
            for tok in p:
                if tok in tok2ix[lang]:
                    vectorized_sentence.append(tok2ix[lang][tok])
                else:
                    """
                    If a token is not found in the vocabulary, we substitute it with the UNK_token index.
                    """
                    vectorized_sentence.append(tok2ix[lang][UNK_token])
            vectorized_pair.append(vectorized_sentence)
        vectorized_pairs.append(vectorized_pair)
    return vectorized_pairs


"""
We vectorize both our training data (`pairs`) and testing data (`test_pairs`) 
using the vectorize_tokenized_pairs function.
"""
pairs = vectorize_tokenized_pairs(pairs, tok2ix)
test_pairs = vectorize_tokenized_pairs(test_pairs, tok2ix)


"""
Although it's not strictly necessary to define a custom class that inherits from Dataset, 
doing so is a common practice in PyTorch. It also ensures consistency across our other tutorials. 
This custom class, `TranslationDataset`, returns a tuple containing a vectorized English sentence 
and its corresponding vectorized French translation.
"""
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0], dtype=torch.long), torch.tensor(self.pairs[idx][1], dtype=torch.long)



"""
Next, we create a DataLoader, which wraps our dataset and allows us to iterate over it, 
fetching 'batch_size' samples at a time. A crucial part of this process is defining a `collate_fn` function. 
This function is responsible for preparing batches of data fetched from `TranslationDataset` so that 
they are suitable for input into our network (which we will define later).

In `collate_fn`, we pad the English and French sequences in each batch separately 
to ensure they are of uniform length. For example:

Given input batch:
    [((Tensor([1,10,2]), Tensor([1,5,8,2])),
     ((Tensor([1,7,11,56,2]), Tensor([1,23,2])),
     ((Tensor([1,110,2]), Tensor([1,66,12,2]))]

The returned batch would be:
    [Tensor([1,10,2,0,0]), Tensor([1,7,11,56,2]), Tensor([1,110,2,0,0])],
    [Tensor([1,5,8,2]), Tensor([1,23,2,0]), Tensor([1,66,12,2])]
"""
def collate_fn(batch):
    src_seqs, trg_seqs = zip(*batch)

    """Find the maximum sequence length in the batch for both source and target languages."""
    src_lens = [len(seq) for seq in src_seqs]
    trg_lens = [len(seq) for seq in trg_seqs]

    max_src_len = max(src_lens)
    max_trg_len = max(trg_lens)

    """Pad sequences to the maximum length found in the batch."""
    src_seqs_padded = [F.pad(seq, (0, max_src_len - len(seq)), value=tok2ix['src_lang'][PAD_token]) for seq in src_seqs]
    trg_seqs_padded = [F.pad(seq, (0, max_trg_len - len(seq)), value=tok2ix['trg_lang'][PAD_token]) for seq in trg_seqs]


    """Stack the padded sequences into tensors."""
    src_seqs_padded = torch.stack(src_seqs_padded)
    trg_seqs_padded = torch.stack(trg_seqs_padded)

    """
    If you want to verify the shapes of the padded sequences, you can print them using:
        print(src_seqs_padded.shape, trg_seqs_padded.shape)
    """

    return src_seqs_padded, trg_seqs_padded

train_dataset = TranslationDataset(pairs)
train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)

"""
While you can also use a DataLoader for test data, we choose not to do so here.
This decision is mainly to demonstrate different coding approaches, 
helping you become comfortable with the variety of styles that different developers might use in real-world projects.
"""


"""
We will now define our machine translation neural network. 
While it is possible to create a single class that inherits from nn.Module 
and handles all the necessary components, we have chosen to use multiple classes. 
This approach demonstrates that defining a neural network using multiple classes 
can enhance readability, maintainability, and facilitate easier testing. 
Each class will represent a distinct component of the network, making the code 
more modular and easier to understand.
"""

"""
A basic sequence-to-sequence (seq2seq) model, like the one we'll use for machine translation, 
consists of two main components: the Encoder and the Decoder. 
Thus, we will define a class named `Encoder` and another class named `Decoder`. 
Since these components need to interact with each other, we'll also define a 
`Seq2Seq` class, which serves as the overarching model that manages the interactions 
between the Encoder and Decoder.
"""


"""
The `Encoder` is an LSTM-based model that processes an English sentence (source language) 
and performs two main tasks:
1. It learns the embeddings of the source language vocabulary (English in this example).
2. It generates vectors that ideally represent the entire input sequence, 
   which in our case is an English sentence.
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


"""
The `Decoder` is responsible for:
1. Learning the embeddings of the target language vocabulary (French in this example).
2. Using the context provided by the source language vectors to generate a translation 
   one token at a time, producing a sequence in the target language.
"""
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, trg, hidden, cell):
        """Add a batch dimension to the target sequence"""
        trg = trg.unsqueeze(1)
        """
           You can uncomment the following lines to check the shape and content of trg:
           print(trg)
           print(trg.shape)
        """
        embedded = self.embedding(trg)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


"""
Finally, we define the `Seq2Seq` class, which integrates the Encoder and Decoder components 
into a complete model. This class handles the flow of data between the encoder and decoder 
and implements the logic for training, including teacher forcing.
"""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
                Arguments:
                src -- A batch of vectorized, padded sequences in the source language (English).
                trg -- A batch of vectorized, padded sequences in the target language (French),
                       where each item is the corresponding translation of the item in src.
                teacher_forcing_ratio -- A ratio that determines the probability of using the actual
                                         target token instead of the predicted one during training.
                """

        """
        Note that the number of samples in src and trg are the same, as each 
        English sentence must have a corresponding French translation.
        """
        batch_size = src.size(0)

        """Determine the maximum length of the target sequences (French) in the batch."""
        trg_len = trg.size(1)

        """Get the size of the target language vocabulary."""
        trg_vocab_size = self.decoder.embedding.num_embeddings


        """
        Initialize a tensor to hold the predictions. This tensor will have dimensions 
        [batch_size, trg_len, trg_vocab_size], where trg_len is the maximum length 
        of the French sentences in the batch.
        """
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        """
        Pass the source sequences (English) through the encoder to obtain the 
        hidden and cell states, which represent the encoded source sentences.
        """
        hidden, cell = self.encoder(src)

        """
        The first input to the decoder is always the <SOS> token.
        """
        input = trg[:, 0]

        for t in range(1, trg_len):
            """
            Pass the previous target token, along with the hidden and cell states from the 
            previous time step, to the decoder to predict the next token.
            """
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            """
            During training, select the token with the highest probability 
            (i.e., the predicted token) from the decoder's output.
            """
            top1 = output.argmax(1)

            """
             Decide whether to use the actual next token from the target sequence (teacher forcing) 
             or the predicted token as the input for the next time step.
             """
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1

        return outputs






"""Instantiate the Encoder, Decoder, and Seq2Seq model, then move the model to the specified device."""
encoder = Encoder(len(tok2ix['src_lang']), 256, 256, 2).to(device)
decoder = Decoder(len(tok2ix['trg_lang']), 256, 256, 2).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

"""
We are now ready to train our machine translation neural network.
"""
epochs = 10
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=tok2ix['trg_lang'][PAD_token])

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for src, trg in train_dataloader:
        src, trg = src.to(device), trg.to(device)

        """Reset the gradients"""
        optimizer.zero_grad()

        """Forward pass: the model predicts the target sequence"""
        output = model(src, trg)

        """Reshape the output and target tensors to fit the criterion"""
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)

        """Flatten the target tensor to match the output shape"""
        trg = trg[:, 1:].contiguous().view(-1)

        """Calculate the loss, backpropagate, and update the weights"""
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        epoch_loss += loss.item()

    """Calculate and print the average loss for this epoch"""
    loss = epoch_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs} Loss: {loss:.4f}')

"""
After training, we will test our model. 
For simplicity, we use a greedy approach to decode the model's predictions. 
More advanced methods, such as beam search, could yield better results but are beyond this scope.
"""

model.eval()
results = []

with torch.no_grad():
    for src_seq, trg_seq in test_pairs:
        """Prepare the source and target sequences"""
        src_seq = torch.tensor(src_seq, dtype=torch.long).unsqueeze(0).to(device)
        trg_seq = torch.tensor(trg_seq, dtype=torch.long).unsqueeze(0).to(device)

        """Get the model's predictions with teacher forcing turned off during evaluation"""
        output = model(src_seq, trg_seq, teacher_forcing_ratio=0)
        output = output.squeeze(0).argmax(1).tolist()

        """Convert the numerical sequences back to tokens for readability"""
        src_tokens = [ix2tok['src_lang'][ix] for ix in src_seq.squeeze(0).tolist() if
                      ix2tok['src_lang'][ix] not in [PAD_token, SOS_token, EOS_token]]
        trg_tokens = [ix2tok['trg_lang'][ix] for ix in trg_seq.squeeze(0).tolist() if
                      ix2tok['trg_lang'][ix] not in [PAD_token, SOS_token, EOS_token]]
        pred_tokens = [ix2tok['trg_lang'][ix] for ix in output if
                       ix2tok['trg_lang'][ix] not in [PAD_token, SOS_token, EOS_token]]

        """Store the source, expected, and predicted sequences for analysis"""
        results.append({'src': ' '.join(src_tokens),'expected': ' '.join(trg_tokens),'predicted': ' '.join(pred_tokens)})

"""Display the results for each example in the test set"""
for i, result in enumerate(results):
    print(f"Example {i + 1}:")
    print(f"Source: {result['src']}")
    print(f"Expected: {result['expected']}")
    print(f"Predicted: {result['predicted']}")
    print('-' * 30)

"""
Sample Output Example:

Example 1:
Source: i'm too busy
Expected: je suis trop occupé
Predicted: je suis trop occupée
------------------------------
"""


"""
Related Tutorials:

Reference [1]: This is a free online course offered by Stanford that provides a comprehensive overview of NLP
and deep learning architectures. It's excellent for understanding theoretical concepts but does not cover
practical implementation or how to use these models with PyTorch.

References [2] and [3]: These sources build upon and complement Reference [1] by providing practical, hands-on
tutorials for implementing seq2seq models. They offer valuable insights into the implementation details that 
are not covered in the theoretical course.

Reference [4]: This tutorial is available on PyTorch's website and is highly valuable. However, it includes
some details that may not be essential for a tutorial focused specifically on seq2seq models. For instance, it
covers preprocessing techniques that might not be necessary for this tutorial. Additionally, it begins with 
attention mechanisms, which could be confusing for those new to seq2seq models. The tutorial also uses separate
optimizers for the encoder and decoder, whereas this tutorial employs a single optimizer for training the entire
end-to-end model.

References: 
[1] - https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4
[2] - https://www.youtube.com/playlist?list=PLqL-7eLmqd9V3faivSAST76YQClS44dSz
[3] - https://www.youtube.com/playlist?list=PLqL-7eLmqd9Vpx4otaVbNeLp-QJiQnOmv
[4] - https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
