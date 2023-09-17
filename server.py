import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import main


class Input(BaseModel):
    data: list[float]


app = FastAPI()

main.load()


@app.get('/')
async def index():
    return HTMLResponse(content=open('index.html').read(), status_code=200)


@app.post("/predict")
async def create_item(input: Input):
    if len(input.data) != 28 * 28 or min(input.data) < 0.0 or max(input.data) > 1.0:
        return 'error'
    y = main.eval(np.array(input.data))[-1]
    return list(y)

