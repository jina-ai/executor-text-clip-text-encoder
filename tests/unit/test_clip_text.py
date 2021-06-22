from jinahub.encoder.clip_text import ClipTextEncoder
from jina import Document, DocumentArray


def test_encode():
    encoder = ClipTextEncoder()
    docs = DocumentArray([
        Document(text=t) for t in ('hello', 'jina')])
    encoder.encode(docs=docs, parameters={})
    for d in docs:
        print(f'{d.embedding.shape}')
