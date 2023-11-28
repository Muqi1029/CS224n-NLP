# NMT(Neural Machine Translation) Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

> we will implement a sequence-to-sequence
(Seq2Seq) network with attention, to build a Neural Machine Translation (NMT) system

## Model description(training process)

 1. In order to apply tensor operations, we must ensure that the sentences in a given batch are of the same length. Thus, we must identify the longest sentence in a batch and pad others to be the same length. Implement the `pad_sents` function in `utils.py`, which shall produce these padded sentences.

```python
    ### YOUR CODE HERE (~6 Lines)
    max_length = max(len(sent) for sent in sents)
    for sent in sents:
        len_sent = len(sent)
        if len_sent < max_length:
            sents_padded.append(sent + [pad_token] * (max_length - len_sent))
        elif len_sent == max_length:
            sents_padded.append(sent)
    ### END YOUR CODE
```

2. Implement the init function in model embeddings.py to initialize the necessary source and target embeddings.

```python
self.source = nn.Embedding(len(vocab.src), embed_size, padding_idx=src_pad_token_idx)
self.target = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=tgt_pad_token_idx)
```

3. Implement the init function in `nmt_model.py` to initialize the necessary model embeddings (using the ModelEmbeddings class from `model_embeddings.py`) and layers(LSTM, projection, and dropout) for the NMT system.

```python
self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)

# these two vectors are used to project final encoded vectors to starting decoder vectors
self.h_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
self.c_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)

# used to compute the weights of attention
self.att_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)

# project the combined output vectors to hidden size vectors
self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)

# convert output vector to the probability of each word in the vocab
self.target_vocab_projection = nn.Linear(hidden_size, len(vocab), bias=False)

self.dropout = nn.Dropout(dropout_rate)
```

4. Implement the encode function in nmt model.py. This function converts the padded source sentences into the tensor X, generates $h_1^{enc}, \vdots, h_m^{enc}$ and computes the initial state $$ and initial cell cdec0 for the Decoder. You can run a non-comprehensive sanity check by executing:

```python
def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
   torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
   """ Apply the encoder to source sentences to obtain encoder hidden states.
       Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

   @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                   b = batch_size, src_len = maximum source sentence length. Note that 
                                   these have already been sorted in order of longest to shortest sentence.
   @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
   @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                   b = batch size, src_len = maximum source sentence length, h = hidden size.
   @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                           hidden state and cell.
   """
   enc_hiddens, dec_init_state = None, None

   ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
   ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
   ###         that there is no initial hidden state or cell for the decoder.
   X = self.model_embeddings.source(source_padded)

   ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
   ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
   ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
   ###         - Note that the shape of the tensor returned by the encoder is (src_len, b, h*2) and we want to
   ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
   pack_X = pack_padded_sequence(X, source_lengths)

   encoded_X, (h_n, c_n) = self.encoder(pack_X)

   # put the batch to the first dimension
   enc_hiddens = pad_packed_sequence(encoded_X)[0].permute(1, 0, 2)

   ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
   ###         - `init_decoder_hidden`:
   ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
   ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
   ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
   ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
   ###         - `init_decoder_cell`:
   ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
   ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
   ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
   ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
   ###
   init_decoder_hidden = self.h_projection(torch.cat((h_n[0], h_n[1]), dim=1))
   init_decoder_cell = self.c_projection(torch.cat((c_n[0], c_n[1]), dim=1))

   dec_init_state = (init_decoder_hidden, init_decoder_cell)
   return enc_hiddens, dec_init_state
```

5. Implement the decode function in `nmt_model.py`, This function constructs $\bar{y}$ and runs the step function over every timestep for the input.

    1. traverse the time dimension of Y 
    2. 


6.  Implement the step function in nmt model.py. This function applies the Decoder’s LSTM cell for a single timestep, computing the encoding of the target subword hdect,the attention scores et, attention distribution αt, the attention output at, and finally the combined output $o_t$. 