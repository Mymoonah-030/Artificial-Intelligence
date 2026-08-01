
# Spam Message Detector (Mini Project)

This is a small ready-made Spam Message Detector project using Python, scikit-learn and CountVectorizer + MultinomialNB.

## What is included
- `data/sms_spam_sample.csv` : Small sample dataset
- `src/train.py` : Script to train the model and save `model.joblib`
- `src/predict.py` : Interactive script to predict a single message using the saved model
- `model.joblib` : (Generated below) trained model
- `requirements.txt` : libraries required

## How to run (recommended in PyCharm)
1. Create a new Project in PyCharm and choose this folder as the project root, or extract the zip and open in PyCharm.
2. Create/choose a Python interpreter (3.8+ recommended).
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model (optional, model is already saved):
   ```bash
   python src/train.py
   ```
5. Run the interactive predictor:
   ```bash
   python src/predict.py
   ```
