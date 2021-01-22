import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

max_length = 250
model = load_model('Models/LSTM_no1n2.h5')

with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


labels = ['Weather/Natural','Sent Mail','Random/NA','Financial/Logistics','Related to Other People',
          'Places','Legal','Business','2-Letter/Random','Other Firms','HR/Recruiting/MBA']

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

while True:
    text = input("Enter the text to classify:\n")
    if text == "end":
        print ("Ending program.")
        break
    elif text == "labels":
        print(labels)
    else:
        new_complaint = [text]
        seq = tokenizer.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=max_length)
        pred = model.predict(padded)
        print(f"{bcolors.BOLD} {bcolors.WARNING}Inputed Text: {text}{bcolors.ENDC} {bcolors.ENDC}")
        print(f"{bcolors.BOLD} {bcolors.WARNING}Email Class: {labels[np.argmax(pred)]}{bcolors.ENDC} {bcolors.ENDC}")