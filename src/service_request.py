from typing import Tuple
import json
import torch
import numpy as np
import requests

SERVICE_URL = "http://localhost:3000/classify"


def sample_random_mnist_data_point() -> Tuple[np.ndarray, np.ndarray]:
    input_tensor_data = torch.tensor([51, 2187, 2, 1941, 6, 39, 1007, 7, 2662, 2, 2197, 3691, 3, 43, 849, 14, 43, 56, 1681, 194,
                            1707, 1105, 1091, 6, 205, 43, 806, 2, 321, 17, 4347, 3, 38, 93, 8, 2, 5099, 4284, 3, 2,
                            1651, 3, 405, 1656, 3, 2052, 3, 7, 8046, 3, 36, 8, 2222, 533, 2, 525, 5, 1971, 1148, 7,
                            2, 588, 20, 4098, 133, 216, 5, 14038, 7, 10002, 12, 8255, 8, 1535, 2, 181, 20, 1351, 7,
                            1524, 6, 32, 28, 2820, 4, 43, 915, 193, 4782, 3, 2749, 3, 6, 35, 42, 9, 359, 155, 8, 2823,
                            2, 525, 5, 39, 5285, 302, 4, 43, 7548, 1404, 6, 255, 39, 63, 2, 1541, 6159, 1005, 3, 86,
                            1122, 104, 1215, 2820, 20, 48, 69, 109, 32, 3528, 3, 39, 93, 97677, 50, 19, 58, 5219, 6,
                            684, 2, 216, 14, 55, 191, 19, 1448, 2898, 5, 837, 7, 14123, 21, 2, 21797, 3325, 97, 6, 90,
                            4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    output_tensor = torch.tensor([0., 1.])

    input_data = input_tensor_data.numpy() 
    expected_output  = output_tensor.numpy()
    return input_data, expected_output


def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text


def main():
    input_data, expected_output = sample_random_mnist_data_point()
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()