import clip
import numpy as np
import torch
from jina import Document, DocumentArray
from jinahub.encoder.clip_text import CLIPTextEncoder


def test_no_documents():
    clip_text_encoder = CLIPTextEncoder()
    test_docs = DocumentArray()
    clip_text_encoder.encode(test_docs)
    assert len(test_docs) == 0  # SUCCESS


def test_clip_batch():
    test_docs = DocumentArray((Document(text='random text') for _ in range(30)))
    clip_text_encoder = CLIPTextEncoder()
    parameters = {'batch_size': 10}
    clip_text_encoder.encode(test_docs, parameters)
    assert 30 == len(test_docs.get_attributes('embedding'))


def test_clip_data():
    docs = []
    words = ['apple', 'banana1', 'banana2', 'studio', 'satelite', 'airplane']
    for word in words:
        docs.append(Document(text=word))

    sentences = ['Jina AI is lit', 'Jina AI is great', 'Jina AI is a cloud-native neural search company', \
                 'Jina AI is a github repo', 'Jina AI is an open source neural search project']
    for sentence in sentences:
        docs.append(Document(text=sentence))

    clip_text_encoder = CLIPTextEncoder()
    clip_text_encoder.encode(DocumentArray(docs), {})

    txt_to_ndarray = {}
    for d in docs:
        txt_to_ndarray[d.text] = d.embedding

    def dist(a, b):
        nonlocal txt_to_ndarray
        a_embedding = txt_to_ndarray[a]
        b_embedding = txt_to_ndarray[b]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satelite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    small_distance = dist('Jina AI is lit', 'Jina AI is great')
    assert small_distance < dist('Jina AI is a cloud-native neural search company', 'Jina AI is a github repo')
    assert small_distance < dist('Jina AI is a cloud-native neural search company',
                                 'Jina AI is an open source neural search project')

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(txt_to_ndarray) == 11
    for text, actual_embedding in txt_to_ndarray.items():
        with torch.no_grad():
            tokens = clip.tokenize(text)
            expected_embedding = model.encode_text(tokens).detach().numpy().flatten()

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)
