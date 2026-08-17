"""
EMBEDDINGS -- turns text (and optionally images) into vectors for
similarity search. Isolated in its own file because "which embedding
model" is a decision independent of chunking, summarization, storage, or
retrieval logic -- swapping the embedding model should never require
touching any other file.

Two embedding strategies are available:

  TEXT embeddings (the default, always used) -- nomic-embed-text via
  Ollama, embeds the SUMMARY of every element (see vision.py/pipeline.py
  for why summaries, not raw content, get embedded). This is what makes
  a chart searchable at all: qwen2.5vl reads the chart and writes a text
  description, and THAT text gets embedded -- the image itself never
  touches the embedding model.

  CLIP embeddings (USE_CLIP=True, opt-in) -- embeds an image's own
  PIXELS directly into the same vector space as text, via a CLIP model.
  See "Why CLIP is NOT used by default" below for why this isn't a
  drop-in upgrade for this pipeline's actual job.

Why CLIP is NOT used by default -- a real evaluation, not a guess:

  CLIP (Contrastive Language-Image Pretraining) is trained to match
  natural photos against short captions -- "a photo of a dog on a
  beach," "a red sports car." It's excellent at broad visual/semantic
  similarity: "find images that LOOK LIKE this description."

  That is a genuinely different capability from what this pipeline
  needs, which is READING a chart -- extracting the discrete facts it
  depicts (Manufacturing added 420 thousand jobs; battery storage fell
  from $280/kWh to $98/kWh). CLIP has no OCR-like capability and no
  concept of "read the axis labels and bar heights" -- it was never
  trained to do that, and embedding a bar chart with CLIP would only
  tell you "this looks like a chart," not what the chart SAYS. This
  pipeline's whole multi-vector design (vision.py's qwen2.5vl summary,
  generation.py's precomputed-totals fix) exists specifically to answer
  questions like "how many jobs were added in manufacturing" -- CLIP
  cannot do that at all, at any accuracy, because it isn't a
  vision-LANGUAGE model that reads and states values; it's a
  vision-similarity model that ranks how alike two things look.

  Verified concretely in this repo's own history (see README's "Real
  bugs this surfaced" #4): the fix that made chart-reading actually work
  was swapping to a stronger VISION-LANGUAGE model (qwen2.5vl:7b), not
  changing the embedding strategy. CLIP addresses a different problem
  than the one that was actually broken.

  CLIP earns its keep for a DIFFERENT kind of question this pipeline
  doesn't currently ask: "find me images that look similar to this
  photo/sketch" -- reverse image search, not fact extraction. It's
  provided here, working and switchable, for exactly that use case, not
  swapped in as a replacement for the vision-language summarization step.
"""

import sys

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"

# Set True to also embed each image's own pixels via CLIP, stored
# alongside (not instead of) the text-summary embedding every element
# already gets. See this file's module docstring for why this is opt-in,
# not a replacement for the vision-model summarization step.
USE_CLIP = False
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def build_text_embeddings() -> Embeddings:
    """The embedding function used for every element's SUMMARY, always --
    this is what vector_store.py's Chroma collection is built with, and
    what a question gets embedded with at query time. Both sides of a
    similarity_search() call MUST use the same embedding model, or
    "closeness" in vector space becomes meaningless.
    """
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


class ClipImageEmbedder:
    """Embeds raw image bytes directly into CLIP's vector space --
    independent of, and NOT interchangeable with, the text embeddings
    above (different model, different vector space, different dimension
    count; a CLIP vector and a nomic-embed-text vector cannot be compared
    to each other at all). Only instantiated when USE_CLIP=True.

    Requires the `transformers` + `torch` packages (see pyproject.toml's
    optional clip extra) -- NOT a dependency of the default pipeline, so
    installing this example doesn't force a multi-GB PyTorch download
    for a feature most runs won't use.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise ImportError(
                "CLIP embeddings need the 'clip' extra: uv pip install -e '.[clip]' "
                "(installs transformers + torch, several GB -- only needed if USE_CLIP=True)"
            ) from e

        print(f"  [embeddings] loading CLIP model {model_name} (one-time, may take a moment)...", file=sys.stderr)
        self._torch = torch
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def embed_image(self, image_bytes: bytes) -> list[float]:
        import io
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features[0].tolist()

    def embed_text(self, text: str) -> list[float]:
        """CLIP has its own text tower, in the SAME space as embed_image --
        this is what makes CLIP retrieval work at all: a text query and
        an image both land in one shared space, unlike nomic-embed-text
        (text-only) plus a vision-model-generated caption (two separate
        embedding steps glued together by the caption text in between).
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with self._torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features[0].tolist()
