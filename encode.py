from jina import Document, DocumentArray, Executor, requests
import torch
import clip

class CLIPTextEncoder(Executor):
    """Encode text into embeddings."""

    def __init__(self, model_name: str = 'ViT-B/32', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, _ = clip.load(model_name, 'cpu')

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        with torch.no_grad():
            for doc in docs:
                input_torch_tensor = clip.tokenize(doc.content)
                embed = self.model.encode_text(input_torch_tensor)
                doc.embedding = embed.cpu().numpy().flatten()
        return docs
