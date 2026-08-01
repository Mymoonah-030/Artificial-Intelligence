# demo_user_input.py
from joblib import load

# Load the pre-trained model
model = load('model.joblib')

print("=== Spam Message Detector ===")
print("Type a message and press Enter (empty to exit)")

while True:
    msg = input("> ")
    if msg.strip() == "":
        print("Exiting...")
        break

    # Prediction
    label = model.predict([msg])[0] #model message → vectorize → probability compare → output

    # Confidence
    if hasattr(model, "predict_proba"): #object, attribute
        proba = model.predict_proba([msg])[0]
        confidence = max(proba)
        print(f"Label: {label}  Confidence: {confidence:.3f}")
    else:
        # If pipeline/classifier does not support predict_proba
        print(f"Label: {label}")

