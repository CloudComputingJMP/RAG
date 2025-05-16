import os


from fastapi import FastAPI
import uvicorn
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel



PROJECT_ID=os.environ.get("PROJECT_ID","aa")
IDX=os.environ.get("IDX",'x')
IDX_EP=os.environ.get("IDX_EP",'x')
IDX_DEP_ID=os.environ.get("IDX_DEP_ID","x")
app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})
aiplatform.init(project=PROJECT_ID, location="europe-central2")
model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

index = aiplatform.MatchingEngineIndex(IDX)
#To use this MatchingEngineIndexEndpoint in another session:
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(IDX_EP)
from  typing import List,Optional
def encode_texts_to_embeddings(sentences: List[str]) -> List[Optional[List[float]]]:
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

@app.get("/recomendations")
def recomendations(query:str,k:int)->List[int]:
    emb=encode_texts_to_embeddings([query])
    resp=index_endpoint.find_neighbors(
        deployed_index_id=IDX_DEP_ID,
        queries=emb,
        num_neighbors=k
    )
    return [int(obj.id) for obj in resp[0]]

@app.get("/create_embedings")
def create_embedings():
    raise NotImplementedError
if __name__ == "__main__":
    uvicorn.run(app, port=5000)
