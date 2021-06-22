import operator
import os
from glob import glob

import clip
import numpy as np
import torch
from PIL import Image
from jina import Flow, Document, DocumentArray, requests, Executor
#from jinahub.encoder.ClipTextEncoder import ClipTextEncoder
import sys

sys.path.insert(1, '../..')

from encode import ClipTextEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_fail():
    class MockExecutor(Executor):
        @requests
        def encode(self, **kwargs):
            pass

    with Flow().add(uses=MockExecutor) as f:
        f.post(on='/test', inputs=[Document(text='whatever')])


def test_clip_text_encoder():
    f = Flow().add(uses={
        'jtype': ClipTextEncoder.__name__,
        'with': {
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        result = f.post(on='/test', inputs=(Document(text='random text') for _ in range(30)),
               return_results=True)
        assert 30 == len(result[0].docs.get_attributes('embedding'))


def test_traversal_path():
    text = 'blah'
    docs = [Document(id='root1', text=text)]
    docs[0].chunks = [Document(id='chunk11', text=text),
                      Document(id='chunk12', text=text),
                      Document(id='chunk13', text=text)
                      ]
    docs[0].chunks[0].chunks = [
                    Document(id='chunk111', text=text),
                    Document(id='chunk112', text=text),
                ]

    f = Flow().add(uses={
        'jtype': ClipTextEncoder.__name__,
        'with': {
            'default_traversal_path': ['c'],
            'model_name': 'ViT-B/32',
        }
    })
    with f:
        result = f.post(on='/test', inputs=docs, return_results=True)
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            print(path)
            assert len(DocumentArray(result[0].data.docs).traverse_flat([path]).get_attributes('embedding')) == count

        result = f.post(on='/test', inputs=docs, parameters={'traversal_path': ['cc']}, return_results=True)
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(result[0].data.docs).traverse_flat([path]).get_attributes('embedding')) == count


def test_no_documents():
    with Flow().add(uses=ClipTextEncoder) as f:
        result = f.post(on='/test', inputs=[], return_results=True)
        assert result[0].status.code == 0  # SUCCESS


def test_clip_data():
    docs = []
    words = ['apple', 'banana1', 'banana2', 'studio', 'satelite', 'airplane']
    for word in words:
        docs.append(Document(text=word))

    sentences = ['Jina AI is lit', 'Jina AI is great', 'Jina AI is a cloud-native neural search company', \
                 'Jina AI is a github repo', 'Jina AI is an open source neural search project']
    for sentence in sentences:
        docs.append(Document(text=sentence))

    with Flow().add(uses=ClipTextEncoder) as f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        txt_to_ndarray = {}
        for d in results[0].docs:
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
    assert small_distance < dist('Jina AI is a cloud-native neural search company', 'Jina AI is an open source neural search project')

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(txt_to_ndarray) == 11
    for text, actual_embedding in txt_to_ndarray.items():
        with torch.no_grad():
            tokens = clip.tokenize(text)
            expected_embedding = model.encode_text(tokens).detach().numpy().flatten()

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)

text = 'blah'
docs = [Document(
    id='root1',
    text=text,
        ),
        Document(id='chunk12', text=text),
        Document(id='chunk13', text=text),
    ]
docs[0].chunks = [Document(id='chunk11', text=text,)]
docs[0].chunks[0].chunks = [Document(id='chunk111', text=text),Document(id='chunk112', text=text),]
docs[0].plot()