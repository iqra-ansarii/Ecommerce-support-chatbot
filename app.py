import os

import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "ecommerce_retrieval_embeddings.npz"
)
RETRIEVAL_DATA_PATH = os.path.join(
    BASE_DIR,
    "ecommerce_retrieval_data.csv.gz"
)


class ONNXEmbeddingModel:

    def __init__(
        self,
        model_path,
        tokenizer_path,
        max_length=256
    ):
        self.tokenizer = Tokenizer.from_file(
            tokenizer_path
        )

        self.tokenizer.enable_truncation(
            max_length=max_length
        )

        self.tokenizer.enable_padding()

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

    def encode(
        self,
        sentences,
        batch_size=32,
        normalize_embeddings=True
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        for start in range(
            0,
            len(sentences),
            batch_size
        ):
            batch = sentences[
                start:start + batch_size
            ]

            encodings = self.tokenizer.encode_batch(
                batch
            )

            input_ids = np.asarray(
                [e.ids for e in encodings],
                dtype=np.int64
            )

            attention_mask = np.asarray(
                [e.attention_mask for e in encodings],
                dtype=np.int64
            )

            token_type_ids = np.asarray(
                [e.type_ids for e in encodings],
                dtype=np.int64
            )

            outputs = self.session.run(
                ["sentence_embedding"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
            )

            embeddings = outputs[0].astype(
                np.float32
            )

            if normalize_embeddings:
                norms = np.linalg.norm(
                    embeddings,
                    axis=1,
                    keepdims=True
                )

                embeddings = embeddings / np.clip(
                    norms,
                    1e-12,
                    None
                )

            all_embeddings.append(
                embeddings
            )

        return np.vstack(
            all_embeddings
        )


# Load ONNX embedding model
embedding_model = ONNXEmbeddingModel(
    MODEL_PATH,
    TOKENIZER_PATH
)


# Load retrieval embeddings
retrieval_train_embeddings = np.load(
    EMBEDDINGS_PATH
)["embeddings"]


# Load retrieval data
retrieval_train_df = pd.read_csv(
    RETRIEVAL_DATA_PATH,
    compression="gzip"
)


def semantic_search(
    query,
    top_k=5
):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    # Both query and retrieval embeddings are normalized,
    # so dot product is equivalent to cosine similarity.
    similarities = np.dot(
        retrieval_train_embeddings,
        query_embedding[0]
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = retrieval_train_df.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarities[
        top_indices
    ]

    return results.reset_index(
        drop=True
    )


def chatbot(
    user_input,
    top_k=1,
    threshold=0.55
):
    results = semantic_search(
        user_input,
        top_k=top_k
    )

    best_result = results.iloc[0]

    score = float(
        best_result["similarity"]
    )

    intent = best_result["intent"]
    response = best_result["response"]

    if score < threshold:
        return {
            "response": (
                "I'm sorry, but I couldn't find enough "
                "information to answer your question. "
                "Could you please rephrase your question?"
            ),
            "intent": "unknown",
            "score": score
        }

    return {
        "response": response,
        "intent": intent,
        "score": score
    }


app = FastAPI(
    title="E-commerce Customer Support Chatbot"
)
app.mount(
    "/static",
    StaticFiles(
        directory=os.path.join(BASE_DIR, "static")
    ),
    name="static"
)

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "index.html")
    )

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    return chatbot(
        request.message
    )