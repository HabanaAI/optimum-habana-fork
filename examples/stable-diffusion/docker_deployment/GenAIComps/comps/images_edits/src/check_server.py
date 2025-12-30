# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import requests

file_name = "/root/edit2509_1.jpg"
endpoint = "http://10.239.15.47:9389/v1/images/edits"
headers = {"accept": "application/json"}

# Prepare the data and files
data = {
    "prompt": "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
}

try:
    with open(file_name, "rb") as image_file:
        files = {"image": (file_name, image_file)}
        response = requests.post(endpoint, headers=headers, data=data, files=files)
        if response.status_code != 200:
            print(f"Failure with {response.reason}!")
        else:
            print(response.json())
except Exception as e:
    print(f"Failure with {e}!")
