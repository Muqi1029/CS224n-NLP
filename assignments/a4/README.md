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

