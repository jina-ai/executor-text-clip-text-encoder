import operator
import os
from glob import glob

import clip
import numpy as np
import torch
from PIL import Image
from jina import Flow, Document, DocumentArray, requests, Executor
from jinahub.encoder.ClipTextEncoder import ClipTextEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_fail():
    class MockExecutor(Executor):
        @requests
        def encode(self, **kwargs):
            pass

    with Flow().add(uses=MockExecutor) as f:
        f.post(on='/test', inputs=[Document(text=np.ones((10,), dtype=np.uint8))])


def test_clip_text_encoder():
    def validate_callback(resp):
        assert 25 == len(resp.docs.get_attributes('embedding'))

    f = Flow().add(uses={
        'jtype': ClipTextEncoder.__name__,
        'with': {
            'default_batch_size': 10,
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(text='random text') for _ in range(25)),
               on_done=validate_callback)


def test_traversal_path():
    def validate_default_traversal(resp):
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count

    def validate_request_traversal(resp):
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count

    text = np.ones((224, 224, 3), dtype=np.uint8)
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


def test_custom_processing():
    f = Flow().add(uses=ClipTextEncoder)
    with f:
        result1 = f.post(on='/test', inputs=[Document(text='random txt')],
                         return_results=True)

    f = Flow().add(uses={
        'jtype': ClipTextEncoder.__name__,
        'with': {
            'use_default_preprocessing': False,
        }
    })

    with f:
        result2 = f.post(on='/test', inputs=[Document(text=np.ones((224, 224, 3), dtype=np.float32))],
                         return_results=True)

    assert result1[0].docs[0].embedding is not None
    assert result2[0].docs[0].embedding is not None
    np.testing.assert_array_compare(operator.__ne__, result1[0].docs[0].embedding, result2[0].docs[0].embedding)


def test_no_documents():
    with Flow().add(uses=ClipTextEncoder) as f:
        results = f.post(on='/test', inputs=[], return_results=True)
        assert results[0].status.code == 0  # SUCCESS


def test_clip_data():
    docs = []
    fruits = ['apple', 'banana']
    for fruit in fruits:
        docs.append(Document(text=fruit))

    with Flow().add(uses=ClipTextEncoder) as f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        txt_name_to_ndarray = {}
        for d in results[0].docs:
            txt_name_to_ndarray[d.id] = d.embedding

    def dist(a, b):
        nonlocal txt_name_to_ndarray
        a_embedding = txt_name_to_ndarray[a]
        b_embedding = txt_name_to_ndarray[b]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')
    assert small_distance < dist('studio', 'satellite')

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(txt_name_to_ndarray) == 5
    for file, actual_embedding in txt_name_to_ndarray.items():
        image = preprocess(Image.open(file)).unsqueeze(0).to('cpu')

        with torch.no_grad():
            expected_embedding = model.encode_image(image).numpy()[0]

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)