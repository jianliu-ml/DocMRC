import torch
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab)

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def get_dep_and_head(long_sentence):
    doc = nlp(long_sentence)
    rel = [token.head.i for token in doc]

    heads = []
    for idx, value in enumerate(rel):
        if value == idx:
            heads.append(value)
    return rel, heads

if __name__ == '__main__':

    text = 'This section collects some of the most common errors you may come across when installing, loading and using spaCy, as well as their solutions. This section collects some of the most common errors you may come across when installing, loading and using spaCy, as well as their solutions. This section collects some of the most common errors you may come across when installing, loading and using spaCy, as well as their solutions.'
    rel, heads = get_dep_and_head(text)
    print(rel, heads)