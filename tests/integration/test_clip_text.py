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
    def validate_callback(resp):
        assert 30 == len(resp.docs.get_attributes('embedding'))

    f = Flow().add(uses={
        'jtype': ClipTextEncoder.__name__,
        'with': {
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(text='random text') for _ in range(30)),
               on_done=validate_callback)


def test_traversal_path():
    def validate_default_traversal(resp):
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count

    def validate_request_traversal(resp):
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count

    text = 'blah'
    docs = [Document(
        id='root1',
        text=text,
        chunks=[
            Document(
                id='chunk11',
                text=text,
                chunks=[
                    Document(id='chunk111', text=text),
                    Document(id='chunk112', text=text),
                ]
            ),
            Document(id='chunk12', text=text),
            Document(id='chunk13', text=text),
        ]
    )]


    f = Flow().add(uses={
        'jtype': ClipTextEncoder.__name__,
        'with': {
            'default_traversal_path': ['c'],
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        f.post(on='/test', inputs=docs, on_done=validate_default_traversal)
        f.post(on='/test', inputs=docs, parameters={'traversal_path': ['cc']}, on_done=validate_request_traversal)


def test_no_documents():
    def validate_response(resp):
        assert resp.status.code == 0  # SUCCESS

    with Flow().add(uses=ClipTextEncoder) as f:
        f.post(on='/test', inputs=[], on_done=validate_response)


def test_clip_data():
    docs = []
    entities = ['apple', 'banana1', 'banana2', 'studio', 'satelite', 'airplane']
    for entity in entities:
        docs.append(Document(text=entity))

    with Flow().add(uses=ClipTextEncoder) as f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        txt_name_to_ndarray = {}
        for d in results[0].docs:
            txt_name_to_ndarray[d.text] = d.embedding

    def dist(a, b):
        nonlocal txt_name_to_ndarray
        a_embedding = txt_name_to_ndarray[a]
        b_embedding = txt_name_to_ndarray[b]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satelite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satelite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satelite')
    assert small_distance < dist('studio', 'satelite')

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(txt_name_to_ndarray) == 6
    # for text, actual_embedding in txt_name_to_ndarray.items():
    #     with torch.no_grad():
    #         tokens = clip.tokenize(text)
    #         expected_embedding = model.encode_text(tokens).detach().numpy()
    #
    #     np.testing.assert_almost_equal(actual_embedding, expected_embedding, 6)
