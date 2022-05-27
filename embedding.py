# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layer.dropout import WordDropout

class Embedding(nn.Module):


    def __init__(self, n_vocab, n_dim, pretrained_vec=None, trainable=True, dp_emb=0., dp_words=0., **kwargs):
        super(Embedding, self).__init__()

        if pretrained_vec is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_vec, freeze=not trainable)
        else:
            self.embedding = nn.Embedding(n_vocab, n_dim, **kwargs)

        self.dp_emb = dp_emb
        if dp_emb > 0:
            self.drop_embedding = nn.Dropout(dp_emb)

        self.dp_words = dp_words
        if dp_words > 0:
            self.drop_words = WordDropout(batch_first=True, p=dp_words)

    def forward(self, input):
        output = self.embedding(input)
        if self.dp_emb > 0:
            output = self.drop_embedding(output)
        if self.dp_words > 0:
            output = self.drop_words(output)

        return output


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3.], requires_grad=True)
    b = torch.zeros_like(a).byte()
    b[0] = 1
    d = a.detach()
    d.zero_()
    c = a**2
    c.sum().backward()

    print(a)
    print(b)
    print(a.grad)