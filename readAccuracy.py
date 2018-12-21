import pickle
from keras.models import load_model
from keras.models import model_from_json

res = pickle.load(open("accuracy.txt", "rb"))

res = [vals for vals in res.values() if len(vals) == 3]
for r in res:
    print(r[0])
    print(r[1])
    print(r[2])