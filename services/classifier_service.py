# from model import get_classifier

# classifier = get_classifier()

# def classify_text(text: str):
#     result = classifier(text)[0]

#     return {
#         "input_text": text,
#         "prediction": result["label"],
#         "confidence": float(result["score"])
#     }


# from model import get_classifier

# classifier = get_classifier()

# def classify_text(text: str):
#     try:
#         if not text.strip():
#             raise ValueError("Input text cannot be empty.")

#         result = classifier(text)[0]

#         return {
#             "input_text": text,
#             "prediction": result["label"],
#             "confidence": float(result["score"])
#         }

#     except Exception as e:
#         return {
#             "error": "Classification failed",
#             "details": str(e)
#         }


import torch
import torch.nn.functional as F
from model import get_model
from label_mapping import label_names

model, tokenizer = get_model()

def classify_text(text: str):

    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class_id].item()

    label = label_names[predicted_class_id]

    
    threshold = 0.65

    if confidence < threshold:
        return {
            "input_text": text,
            "prediction": "out_of_scope",
            "confidence": round(confidence, 4)
        }

    return {
        "input_text": text,
        "prediction": label,
        "confidence": round(confidence, 4)
    }