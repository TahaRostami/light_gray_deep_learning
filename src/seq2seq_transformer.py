"""
This tutorial demonstrates how to train a transformer model for machine translation using PyTorch's
`nn.Transformer`,with options for decoding via either a beam search or greedy approach.
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from torch import Tensor
from torch.nn import Transformer

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

"""
    <PAD>: Padding token used to ensure that all sequences in a batch have the same length.
     For <PAD> we also store its index in a separate variable because in our code we need
     to access and use this particular variable many time. For other special tokens I do not
     do this.
"""
PAD_token, PAD_IDX = "<PAD>", 0

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
    with open("../data/eng-fra.txt", 'r', encoding='latin-1') as f:
        for line in f.readlines():
            if n is not None and n == len(pairs):
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
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch.to(device), trg_batch.to(device)


dataset = TranslationDataset(pairs)
dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)


"""
In this section, we will define our machine translation neural network. Although it's possible to encapsulate
all necessary components within a single class that inherits from nn.Module, we've opted for a modular approach
by defining multiple classes. This method improves readability, maintainability, and testing flexibility. Each
class encapsulates a specific component of the network, making the code more modular and easier to understand.
"""

"""
We define three distinct classes for our model:
    1- TokenEmbedding: Responsible for converting a tensor of input indices into their corresponding tensor
                        of embeddings.
    2- PositionalEncoding: PositionalEncoding: Adds positional information to the input tokens, helping the
                            model understand the order of words.
    3- Seq2SeqTransformer: The core machine translation model that integrates the above components to function
                            together.

Our implementation is inspired by Reference [1]. The main difference is that Reference [1] uses batch_first=False,
while we've modified the implementation to use batch_first=True to maintain consistency across our tutorials.
Additionally, we've introduced an option to enable or disable positional encoding, allowing you to experiment with
and assess its impact on the model's performance.
"""


"""The TokenEmbedding class is straightforward—it converts a tensor of input indices into a tensor of embeddings."""
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """tokens should be (batch_size, seq_len)"""
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


"""
The PositionalEncoding class provides positional information to the input tokens. By default, it adds 
positional encoding to the embeddings. However, we can disable this feature to explore its effects.
"""
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000,pos_enc_is_enb=True):
        super(PositionalEncoding, self).__init__()

        """
        By default, pos_enc_is_enb=True, meaning positional encoding is applied. 
        If set to False, no positional encoding will be added to the embeddings.

        Exercise: Consider implementing alternative positional encodings, 
        such as a learnable positional encoding.
        """
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        """Shape: (1, maxlen, emb_size)"""
        pos_embedding = pos_embedding.unsqueeze(0)

        self.pos_enc_is_enb=pos_enc_is_enb

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        """Ensure token_embedding.shape is (batch_size, seq_len, emb_size)"""
        if self.pos_enc_is_enb:
            seq_len = token_embedding.size(1)
            return self.dropout(token_embedding + self.pos_embedding[:, :seq_len, :])
        else:
            return self.dropout(token_embedding)


"""
The Seq2SeqTransformer class is our main machine translation model, which combines the token embedding, 
positional encoding, and a Transformer architecture.
"""
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int,
                 src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1,pos_enc_is_enb=True):
        super(Seq2SeqTransformer, self).__init__()
        """Ensure batch_first=True"""
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                       dropout=dropout, batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout,pos_enc_is_enb=pos_enc_is_enb)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        """src and trg should have shape (batch_size, seq_len)"""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))


        outs = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        """src should have shape (batch_size, seq_len)"""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        """tgt should have shape (batch_size, seq_len)"""
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)



"""
In training, we use a subsequent word mask to prevent the model from considering future words when generating predictions.
"""

def generate_square_subsequent_mask(sz):
    """
        Reference [1] uses the following code for square_subsequent_mask (also you may hear lookahead_mask).
        However, this code led to numerical instability in my dataset:
            mask = torch.triu(torch.ones((sz, sz), device=device)) == 1
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
        Instead, I use boolean masking, which is more stable.
        """

    mask = torch.triu(torch.ones(sz, sz), diagonal=1).to(device)
    return mask == 1


"""
We also define a function to generate the necessary masks for the source and target sequences.
This function ensures that all masks are consistently of type torch.bool, which helps prevent
type mismatch warnings during training.
"""
def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).type(torch.bool).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask






torch.manual_seed(0)

"""Instantiate the Seq2SeqTransformer model and move it to the specified device."""
transformer = Seq2SeqTransformer(3, 3, 512,
                                 4, len(tok2ix['src_lang']), len(tok2ix['trg_lang']), 512,pos_enc_is_enb=True)

"""Initialize model parameters using Xavier uniform initialization for better convergence, got from Reference [1]."""
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = transformer.to(device)


"""We are now ready to train our machine translation neural network."""

"""Define loss function, ignoring the padding index in the target sequences."""
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

"""Set up the optimizer with learning rate and other hyperparameters, got from Reference [1]."""
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for src, tgt in dataloader:

        """Reset the gradients"""
        optimizer.zero_grad()

        """Move the batch of sequences to the device"""
        src = src.to(device)
        tgt = tgt.to(device)

        """Use the entire batch, excluding the last token in the target sequence."""
        tgt_input = tgt[:, :-1]

        """ Generate masks for the source and target sequences."""
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        """Pass source and target sequences, along with their masks, through the model."""
        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        """The ground truth sequence shifted by one position."""
        tgt_out = tgt[:, 1:]

        """
           Reshape the output to match the expected dimensions in the criterion.
           Shape: [batch_size * (tgt_seq_len - 1), vocab_size]
        """
        output = output.view(-1, output.shape[-1])

        """Shape: [batch_size * (tgt_seq_len - 1)]"""
        tgt_out = tgt_out.reshape(-1)

        """Compute the loss and perform backpropagation"""
        loss = criterion(output, tgt_out)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

    """Calculate and print the average loss for this epoch"""
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')


"""After training, we will test our model using a decoding approach."""

"""Decode the model's predictions using a beam search approach."""
def beam_search_decode(model, src, beam_width=5, max_len=50):
    model.eval()
    with torch.no_grad():
        """Add batch dimension"""
        src = src.unsqueeze(0).to(device)
        src_mask = torch.zeros((src.size(1), src.size(1)), device=device).type(torch.bool)
        src_padding_mask = (src == PAD_IDX)

        """Encode the source sequence"""
        memory = model.encode(src, src_mask)
        batch_size = src.size(0)

        """Initialize beams"""

        hypotheses = [[tok2ix['trg_lang'][SOS_token]]]
        scores = torch.zeros(1, beam_width).to(device)
        completed_hypotheses = []

        for _ in range(max_len):
            new_hypotheses = []
            new_scores = []

            for hyp_idx, hyp in enumerate(hypotheses):
                hyp_tensor = torch.tensor(hyp, dtype=torch.long).unsqueeze(0).to(device)
                tgt_mask = generate_square_subsequent_mask(hyp_tensor.size(1)).type(torch.bool).to(device)
                tgt_padding_mask = (hyp_tensor == PAD_IDX)

                """Decode the current hypothesis"""
                output = model.decode(hyp_tensor, memory, tgt_mask)
                output = model.generator(output)

                """Get the next token probabilities"""
                topk_scores, topk_indices = output[:, -1, :].topk(beam_width, dim=1)

                for k in range(beam_width):
                    new_hyp = hyp + [topk_indices[0, k].item()]
                    new_hypotheses.append(new_hyp)
                    new_scores.append(scores[0, hyp_idx] + topk_scores[0, k].item())

            """Select top-k hypotheses based on scores"""
            if not new_hypotheses:
                break

            topk_scores, topk_indices = torch.tensor(new_scores).topk(beam_width)
            hypotheses = [new_hypotheses[i] for i in topk_indices]
            scores = topk_scores.unsqueeze(0).to(device)

            """Collect completed hypotheses"""
            for hyp in hypotheses[:]:
                if hyp[-1] == tok2ix['trg_lang'][EOS_token]:
                    completed_hypotheses.append(hyp)
                    hypotheses.remove(hyp)

            if not hypotheses:
                break

        """Return the best hypothesis"""
        if completed_hypotheses:
            best_hyp = max(completed_hypotheses, key=lambda hyp: sum(scores[0, hypotheses.index(hyp)].item() for hyp in hypotheses))
        else:
            best_hyp = hypotheses[0]

        pred_tokens = [ix2tok['trg_lang'][idx] for idx in best_hyp if
                       idx not in [tok2ix['trg_lang'][PAD_token], tok2ix['trg_lang'][SOS_token],
                                   tok2ix['trg_lang'][EOS_token]]]
        return pred_tokens


"""Decode the model's predictions using a greedy approach."""
def greedy_decode(model, src, max_len=50):
    model.eval()
    with torch.no_grad():
        """Add batch dimension"""
        src = src.unsqueeze(0)
        """Generate masks"""
        src_mask, _, src_padding_mask, _ = create_mask(src, src)
        """Encode the source sequence"""
        memory = model.encode(src, src_mask)

        """Start decoding"""
        trg_indices = [tok2ix['trg_lang'][SOS_token]]
        for _ in range(max_len):
            trg_tensor = torch.tensor(trg_indices, dtype=torch.long).unsqueeze(0).to(device)
            trg_mask = generate_square_subsequent_mask(trg_tensor.size(1)).to(device)
            trg_padding_mask = (trg_tensor == PAD_IDX).to(device)

            output = model.decode(trg_tensor, memory, trg_mask)
            output = model.generator(output)

            next_token = output.argmax(2)[:, -1].item()
            trg_indices.append(next_token)
            if next_token == tok2ix['trg_lang'][EOS_token]:
                break

        pred_tokens = [ix2tok['trg_lang'][idx] for idx in trg_indices if idx not in [tok2ix['trg_lang'][PAD_token], tok2ix['trg_lang'][SOS_token], tok2ix['trg_lang'][EOS_token]]]
        return pred_tokens


"""You can replace the default decoding method with beam_search_decode if you prefer."""
decoding_approach=greedy_decode
for source, expected in test_pairs:
    source_tensor = torch.tensor(source, dtype=torch.long).to(device)
    translated_tokens = decoding_approach(model, source_tensor)

    source_tokens = [ix2tok['src_lang'][idx] for idx in source if idx not in [tok2ix['src_lang'][PAD_token], tok2ix['src_lang'][SOS_token], tok2ix['src_lang'][EOS_token]]]
    expected_tokens = [ix2tok['trg_lang'][idx] for idx in expected if idx not in [tok2ix['trg_lang'][PAD_token], tok2ix['trg_lang'][SOS_token], tok2ix['trg_lang'][EOS_token]]]

    print("Source: ", ' '.join(source_tokens))
    print("Expected: ", ' '.join(expected_tokens))
    print("Translated: ", ' '.join(translated_tokens))
    print('-' * 30)

"""
Example Output:

Source:  i'm too busy
Expected:  je suis trop occupã©
Translated:  je suis trop occupã©
------------------------------
Source:  is she all right
Expected:  <UNK> bien 
Translated:  est-elle bien 
------------------------------
Source:  i could help
Expected:  je pourrais aider
Translated:  je pouvais aider
------------------------------
Source:  i didn't see you
Expected:  je ne vous ai pas vues
Translated:  je ne vous ai pas vu
"""

"""
Related Tutorials:


Reference [1]: PyTorch’s tutorial for translation transformers. Our implementation is inspired by this tutorial,
with modifications to use batch_first=True and to include beam search.

Reference [2]: A tutorial on time series prediction using nn.Transformer.

References [3] [4]: tutorials that demonstrate implementing a Transformer model from scratch 
                    (i.e., without using nn.Transformer).

Reference [5]: A free online course by Stanford offering a deep dive into NLP and deep learning architectures.


References: 
[1] - https://pytorch.org/tutorials/beginner/translation_transformer.html
[2] - https://www.youtube.com/playlist?list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi
[3] - https://www.youtube.com/playlist?list=PLqL-7eLmqd9V3faivSAST76YQClS44dSz
[4] - https://www.youtube.com/playlist?list=PLqL-7eLmqd9Vpx4otaVbNeLp-QJiQnOmv
[5] - https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4

"""