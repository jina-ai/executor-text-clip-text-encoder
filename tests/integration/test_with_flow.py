from jina import Document, DocumentArray, Flow, requests, Executor
from jinahub.encoder.clip_text import CLIPTextEncoder


def test_fail():
    class MockExecutor(Executor):
        @requests
        def encode(self, **kwargs):
            pass

    with Flow().add(uses=MockExecutor) as f:
        f.post(on='/test', inputs=[Document(text='whatever')])


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
        'jtype': CLIPTextEncoder.__name__,
        'with': {
            'default_traversal_paths': ['c'],
            'model_name': 'ViT-B/32',
        }
    })
    with f:
        result = f.post(on='/test', inputs=docs, return_results=True)
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            assert len(DocumentArray(result[0].data.docs).traverse_flat([path]).get_attributes('embedding')) == count

        result = f.post(on='/test', inputs=docs, parameters={'traversal_paths': ['cc']}, return_results=True)
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(result[0].data.docs).traverse_flat([path]).get_attributes('embedding')) == count


def test_no_documents():
    with Flow().add(uses=CLIPTextEncoder) as f:
        result = f.post(on='/test', inputs=[], return_results=True)
        assert result[0].status.code == 0  # SUCCESS
