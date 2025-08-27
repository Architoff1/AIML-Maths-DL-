import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

gates = {
    "AND":{
        "inputs": np.array([[0,0],[0,1],[1,0],[1,1]]),
        "outputs": np.array([[0],[0],[0],[1]])
    },
    "OR":{
        "inputs": np.array([[0,0],[0,1],[1,0],[1,1]]),
        "outputs": np.array([[0],[1],[1],[1]])
    },
    "XOR":{
        "inputs": np.array([[0,0],[0,1],[1,0],[1,1]]),
        "outputs": np.array([[0],[1],[1],[0]])
    },
    "NOR":{
        "inputs": np.array([[0,0],[0,1],[1,0],[1,1]]),
        "outputs": np.array([[1],[0],[0],[0]])
    },
    "NAND":{
        "inputs": np.array([[0,0],[0,1],[1,0],[1,1]]),
        "outputs": np.array([[1],[1],[1],[0]])
    }
}

def create_ann():
    model = Sequential([
        Dense(4,input_dim=2,activation='relu'),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics="accuracy")
    return model

#Train
models={}

for gate,data in gates.items():
    print(f"Training model for {gate} gate...")
    model = create_ann()
    model.fit(data["inputs"], data["outputs"], epoch=500, verbose=0)
    models[gate] = model
    print(f"Model for {gate} trainedl.\n")

#Test
def test_gate(model, gate_name):
    print(f"Testing {gate_name} gate:")
    predictions = (model.predict(gates[gate_name]["inputs"]) > 0.5).astype(int)
    for inp,pred in zip(gates[gate_name]["inputs"], predictions):
        print(f"Input: {inp} => Outputs: {pred[0]}")
    print()

for gate, model in models.items():
    test_gate(model,gate)