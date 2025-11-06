import cv2
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


import pickle

def load_all_pickles(file_path):
    data = []
    with open(file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break  # Chegou no fim do arquivo
    return data

# Exemplo de uso:
features_list = load_all_pickles("features.pkl")
labels_list = load_all_pickles("labels.pkl")

# Se forem listas
features = [item for sublist in features_list for item in sublist]
labels = [item for sublist in labels_list for item in sublist]

features = [f for f in features if len(f) == 96]
labels = labels[:len(features)]
features = np.array(features)

features = np.array(features)
labels = np.array(labels)

ratio = 1.5
class0_idx = np.where(labels == 0)[0]
class1_idx = np.where(labels == 1)[0]

if len(class1_idx) > len(class0_idx) * ratio:
    class1_idx = np.random.choice(class1_idx, round(len(class0_idx) * ratio), replace=False)

indices = np.concatenate([class0_idx, class1_idx])
np.random.shuffle(indices)

dataset_balanced = features[indices]
labels_balanced = labels[indices]

class0_idx = np.where(labels == 0)[0]
class1_idx = np.where(labels == 1)[0]

"""
# pegar apenas o mesmo número de exemplos da classe 1
np.random.shuffle(class1_idx)
ratio = 3
class1_idx = class1_idx[:len(class0_idx) * ratio]


selected_idx = np.concatenate([class0_idx, class1_idx])
np.random.shuffle(selected_idx)

dataset_balanced = features[selected_idx]
labels_balanced = labels[selected_idx]"""


print("Exemplos classe 0:", np.sum(labels_balanced==0))
print("Exemplos classe 1:", np.sum(labels_balanced==1))




clf = MLPClassifier(
    hidden_layer_sizes=(512,),
    activation="relu",
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    batch_size=64,
    random_state=42,
    alpha=0.001,
    early_stopping=True,
    validation_fraction=0.1
)

clf.fit(dataset_balanced, labels_balanced)
    
# Parâmetros
n = 200

def compute_color_histogram(img_bgr, bins=32):
    """
    Calcula histogramas separados para B, G e R,
    e concatena todos em um vetor único.
    """
    chans = cv2.split(img_bgr)
    hist_features = []
    for chan in chans:
        hist, _ = np.histogram(chan, bins=bins, range=(0, 256))
        hist_features.extend(hist)
    return np.array(hist_features)


# Abrir vídeo
video = cv2.VideoCapture("Video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
while True:
    ret, frame = video.read()
    if not ret:
        break

    output_frame = frame.copy()

    for i in range(0, frame.shape[1], n):
        for j in range(0, frame.shape[0], n):
            window_img = frame[j:j+n, i:i+n, :]
            hist_features = compute_color_histogram(window_img)
            prediction = clf.predict([hist_features])[0]

            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
            cv2.rectangle(output_frame, (i, j), (i+n, j+n), color, 2)

    cv2.imshow("Classificação", output_frame)
    
    if cv2.waitKey(delay) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
