import csv
import pandas as pd
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec

WORD2VEC = "GoogleNews-vectors-negative300.bin"
GLOVE = "glove.6B.300d.txt"
DEPS = "deps.words.txt"

TEST_FILE = "word-test.v1.txt"
TESTS = [
    "capital-world",
    "currency",
    "city-in-state",
    "family",
    "gram1-adjective-to-adverb",
    "gram2-opposite",
    "gram3-comparative",
    "gram6-nationality-adjective"
]

def loadword2vecbin(file):
    return Word2Vec.load_word2vec_format(file, binary=True)

def word2vectodf(model):
    words = {}
    for word in model.vocab:
        words[word] = model[word]
    df = pd.DataFrame(words).T
    # okay, turns out saving this is not a good idea; re-converting it every time is easier
    # df.to_csv("GoogleNews-vectors-negative300.txt", sep=" ", header=None, quoting=csv.QUOTE_NONE)
    return df

def loadembeddings(file):
    return pd.read_csv(file, delim_whitespace=True, header=None, index_col=0, quoting=csv.QUOTE_NONE)

def runtests(df, file=TEST_FILE, tests=TESTS, normalize=False, lowercase=False):
    dfnorms = norm(df, axis=1)
    with open(file, "r", newline="") as f:
        testname = None
        usetest = False
        results = {}
        result = None
        for line in f:
            if line.startswith(":"):
                testname = line[2:].rstrip("\n")
                usetest = testname in tests
                if usetest:
                    if result is not None:
                        print(result)
                    print(testname)
                    result = {"correct":0, "total":0}
                    results[testname] = result
            elif usetest:
                words = line.lower().split() if lowercase else line.split()
                best = findbest(df, words[0], words[1], words[2], dfnorms, normalize)
                if words[3] == best:
                    result["correct"] += 1
                result["total"] += 1
    return results

def findbest(df, a, b, c, dfnorms=None, normalize=False):
    va = df.loc[a]
    vb = df.loc[b]
    vc = df.loc[c]
    v = vb/norm(vb) - va/norm(va) + vc/norm(vc) if normalize else vb - va + vc
    cossims = df.dot(v) / (dfnorms if dfnorms is not None else norm(df, axis=1)) # norm(v) factors out of the argmax
    cossims.loc[[a,b,c]] = None # exclude from argmax
    return cossims.idxmax()
