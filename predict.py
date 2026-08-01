
import joblib, os
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE, 'model.joblib')

model = joblib.load(MODEL_PATH)

def predict(message):
    pred = model.predict([message])[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = max(model.predict_proba([message])[0])
    return pred, proba

if __name__ == '__main__':
    print('Type a message and press Enter (empty to exit)')
    while True:
        msg = input('> ').strip()
        if not msg:
            break
        label, conf = predict(msg)
        if conf is not None:
            print(f'Label: {label}  Confidence: {conf:.3f}')
        else:
            print(f'Label: {label}')


'''
predict.py = model load করে, prediction দেয় (function হিসেবে)

def predict(message):
    pred = model.predict([message])[0]


– এটি function হিসেবে prediction return করে, যেন অন্য Python file থেকে সহজে call করা যায়।
'''