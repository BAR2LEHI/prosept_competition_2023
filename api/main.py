from typing import Annotated, List

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model import model_job


origins = [
    "http://proseptmatching.zapto.org:3000",
    "http://proseptmatching.zapto.org:8080",
    "http://proseptmatching.zapto.org:8001",
    "http://proseptmatching.zapto.org:8000",
    "http://proseptmatching.zapto.org:5173",
    "http://proseptmatching.zapto.org",
    "http://ds_ml.:8001"
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
