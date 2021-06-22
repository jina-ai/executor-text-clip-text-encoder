from jina import DocumentArray, Executor, requests
import torch
import clip
from typing import Iterable, Optional, List


def _batch_generator(data: DocumentArray, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


class CLIPTextEncoder(Executor):
    """Encode text into embeddings."""

    def __init__(self,
                 model_name: str = 'ViT-B/32',
                 default_batch_size: int = 32,
                 default_traversal_paths: List[str] = ['r'],
                 default_device: Optional[str] = 'cpu',
                 jit: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = default_device
        self.model, _ = clip.load(model_name, 'cpu', jit)
        self.default_traversal_paths = default_traversal_paths
        self.default_batch_size = default_batch_size

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        traversal_paths = parameters.get('default_traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('default_batch_size', self.default_batch_size)

        # traverse through all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without text
        filtered_docs = [doc for doc in flat_docs if doc.text is not None]

        return _batch_generator(filtered_docs, batch_size)

    def _create_embeddings(self, document_batches_generator: Iterable):
        with torch.no_grad():
            for document_batch in document_batches_generator:
                text_batch = [d.text for d in document_batch]
                tensor = clip.tokenize(text_batch)
                tensor = tensor.to(self.device)
                embedding_batch = self.model.encode_text(tensor)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(document_batch, numpy_embedding_batch):
                    document.embedding = numpy_embedding
