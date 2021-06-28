<p align="center">
<img src="https://github.com/jina-ai/jina/blob/master/.github/logo-only.gif?raw=true" alt="Jina banner" width="200px">
</p>

# CLIPTextEncoder

 **CLIPTextEncoder** is a class that wraps the text embedding functionality from the **CLIP** model.

The **CLIP** model was originally proposed in  [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`CLIPTextEncoder` encodes data from a `np.ndarray` of strings and returns a `np.ndarray` of floating point values.

- Input shape: `BatchSize `

- Output shape: `BatchSize x EmbeddingDimension`

## Prerequisites

None

## Encode with the encoder:

The following example shows how to generate output embeddings given an input `np.ndarray` of strings.

```python
# Input data
text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

# Encoder embedding 
encoder = CLIPTextEncoder()
embeddeding_batch_np = encoder.encode(text_batch)


## Usages

### Via JinaHub (🚧W.I.P.)

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
        uses='jinahub+docker://CLIPTextEncoder',
        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CLIPTextEncoder'
    volumes: '/your_home_folder/.cache/clip:/root/.cache/clip'
```


### Via Pypi

1. Install the `jinahub-text-clip-text-encoder`

	```bash
	pip install git+https://github.com/jina-ai/executor-text-clip-text-encoder.git
	```

1. Use `jinahub-text-clip-text-encoder` in your code

	```python
	from jinahub.encoder.clip_text import CLIPTextEncoder
	from jina import Flow
	
	f = Flow().add(uses=CLIPTextEncoder)
	```


### Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-clip-text-encoder.git
	cd executor-text-CLIP
	docker build -t jinahub-clip-text .
	```

1. Use `jinahub-clip-text` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(
	        uses='docker://jinahub-clip-text:latest',
	        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
	```
	


## Example 


```python
from jina import Flow, Document
import numpy as np
	
f = Flow().add(
        uses='jinahub+docker://CLIPTextEncoder',
        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
	
def check_emb(resp):
    for doc in resp.data.docs:
        if doc.emb:
            assert doc.emb.shape == (512,)
	
with f:
	f.post(
	    on='/foo', 
	    inputs=Document(text='your text'), 
	    on_done=check_emb)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the [`text`](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes) attribute.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the `embedding` attribute filled with an `ndarray` of the shape `512` with `dtype=float32`.



## Reference
- https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
- https://github.com/openai/CLIP