import copy

from jina import Document, DocumentArray, Flow, requests, Executor
from jinahub.encoder.clip_text import CLIPTextEncoder


def test_fail():
    class MockExecutor(Executor):
        @requests
        def encode(self, **kwargs):
            pass

    MockExec = MockExecutor()
    MockExec.encode(inputs=[Document(text='whatever')])


def test_traversal_path():
    text = 'blah'
    docs = DocumentArray([Document(id='root1', text=text)])
    docs[0].chunks = [Document(id='chunk11', text=text),
                      Document(id='chunk12', text=text),
                      Document(id='chunk13', text=text)
                      ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', text=text),
        Document(id='chunk112', text=text),
    ]

    encoder = Executor.load_config('''
        jtype: CLIPTextEncoder
        with: 
           default_traversal_paths: ['c']
           model_name: ViT-B/32
        ''')

    original_docs = copy.deepcopy(docs)
    encoder.encode(docs=docs, parameters={}, return_results=True)
    for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
        assert len(docs.traverse_flat([path]).get_attributes('embedding')) == count

    encoder.encode(docs=original_docs, parameters={'traversal_paths': ['cc']}, return_results=True)
    for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
        assert len(original_docs.traverse_flat([path]).get_attributes('embedding')) == count
