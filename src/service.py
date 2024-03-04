# import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

lstm_runner = bentoml.pytorch.get("lstm_model:latest").to_runner()
svc = bentoml.Service('lstm_classifier',runners=[lstm_runner])

# class CustomerComment(BaseModel):
#     comment: str

# input_spec = JSON(pydantic_model=CustomerComment)
# print("Pass")

@svc.api(input=JSON(),output=JSON())
def classify(input_data) -> JSON:
    print("Hello!: For Debug")
    dummy_response = {"result": "dummy_classification", "confidence": 0.9}
    return dummy_response
    # result = lstm_runner.run(input_data)
    # return (result)



