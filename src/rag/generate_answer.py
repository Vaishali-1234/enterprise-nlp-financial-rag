import requests

def generate_answer(prompt):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    options={
    "num_predict": 200
}
    response = requests.post(url, json=payload)
    return response.json()["response"]