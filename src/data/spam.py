import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def prepare_data(test=False):
    data = pd.read_csv("data/sms_train.csv")
    y = data["label"]

    tmp = data["message"].tolist()
    tmp = [re.sub(r"[^\w]", " ", s) for s in tmp]
    documents = [re.sub(r" +", " ", s) for s in tmp]

    vectorizer = CountVectorizer(strip_accents="unicode", lowercase=True)
    vectorizer.fit(documents)

    term_doc_matrix = vectorizer.transform(documents).toarray()

    terms = vectorizer.get_feature_names_out()

    X = pd.DataFrame(term_doc_matrix, columns=terms)
    col = X.columns

    X = X.to_numpy()
    y = y.to_numpy().ravel()

    if not test:
        return X, y

    else:
        data = pd.read_csv("data/sms_test.csv")

        tmp = data["message"].tolist()
        tmp = [re.sub(r"[^\w]", " ", s) for s in tmp]
        new_documents = [re.sub(r" +", " ", s) for s in tmp]

        new_term_doc_matrix = vectorizer.transform(new_documents).toarray()

        X = pd.DataFrame(new_term_doc_matrix, columns=terms)

        X = X.to_numpy()

        return X
