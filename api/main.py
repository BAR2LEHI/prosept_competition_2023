from typing import Annotated, List

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model import model_job


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8001",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:1",
    "http://localhost:80",
    "http://127.0.0.1:80",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/machine-matching/')
def machine_matching(
    name_dealer_product: Annotated[str, Body(embed=True)]
) -> List[int]:
    result = model_job(name_dealer_product)
    return result[0]
