import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

n = 200

def compute_histogram(img_bgr, bins=32):
    # Converte para RGB (OpenCV usa BGR por padrão)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Calcula histogramas separados para R, G e B
    hist_r = np.histogram(img_rgb[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(img_rgb[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(img_rgb[:, :, 2], bins=bins, range=(0, 256))[0]

    # Normaliza e concatena os três canais
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    hist /= np.sum(hist) + 1e-6  # evita divisão por zero
    return hist


def mouse_click(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_RBUTTONDOWN:
        clicked_points.append((x, y, 'right'))  # só botão direito agora

clicked_points = [] 
dataset = []
labels = []
treined = False
meta_info = []

clf = MLPClassifier(
    hidden_layer_sizes=512,
    activation="relu",
    solver='adam',
    learning_rate_init=0.001,
    alpha=0.0005,
    batch_size=16,
    max_iter=50,
    warm_start=True,
    shuffle=True,
    random_state=42,
    early_stopping=False
)

cv2.namedWindow("test")
cv2.setMouseCallback("test", mouse_click)

video = cv2.VideoCapture('Video.mp4')
frame_id = 0

while True:
    ret, base_frame = video.read()
    if not ret:
        break

    frame = base_frame.copy()

    # ⚡️ Todos os blocos agora começam como classe 1 (verde)
    default_dataset = []
    default_labels = []
    for i in range(0, frame.shape[1], n):
        for j in range(0, frame.shape[0], n):
            window_img = base_frame[j:j + n, i:i + n, :]
            hist_features = compute_histogram(cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY))
            default_dataset.append(hist_features)
            default_labels.append(1)  # classe 1 por padrão
            cv2.rectangle(frame, (i, j), (i + n, j + n), (0, 255, 0), 1)

    dataset = default_dataset.copy()
    labels = default_labels.copy()

    while True:
        show_frame = frame.copy()

        # se já tiver modelo treinado, colorir de acordo
        if treined:
            for i in range(0, show_frame.shape[1], n):
                for j in range(0, show_frame.shape[0], n):
                    window_img = show_frame[j:j + n, i:i + n, :]
                    hist_features = compute_histogram(cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY))
                    prediction = clf.predict([hist_features])
                    color = (0, 255, 0) if prediction[0] == 1 else (0, 0, 255)
                    cv2.rectangle(show_frame, (i, j), (i + n, j + n), color, 2)

        # aplicar cliques (classe 0)
        for x, y, label in clicked_points:
            x_window = x // n
            y_window = y // n
            idx = y_window * (frame.shape[1] // n) + x_window
            labels[idx] = 0  # altera o label desse bloco para classe 0

            y1, y2 = y_window * n, (y_window + 1) * n
            x1, x2 = x_window * n, (x_window + 1) * n
            show_frame[y1:y2, x1:x2, :] = ((show_frame[y1:y2, x1:x2, :] + [0, 0, 255]) // 2).astype(np.uint8)

        cv2.imshow("test", show_frame)

        key = cv2.waitKey(1)
        if key == ord('n'):
            # Salvar as features e labels atuais (com os blocos alterados)
            with open("features.pkl", "ab") as f:
                pickle.dump(dataset, f)
            with open("labels.pkl", "ab") as f:
                pickle.dump(labels, f)

            print(f"Frame {frame_id} salvo com {len(dataset)} blocos.")
            
            clicked_points = []
            break
        if key == ord('p'):
            clicked_points = []
            break

        if key == ord('t'):
            PATH_FEATURES = "features.pkl"
            PATH_LABELS = "labels.pkl"
            PATH_MODEL = "mlp_model.pkl"

            with open(PATH_FEATURES, 'wb') as f:
                pickle.dump(dataset, f)
            with open(PATH_LABELS, 'wb') as f:
                pickle.dump(labels, f)

            clf.fit(dataset, labels)
            treined = True

            with open(PATH_MODEL, "wb") as f:
                pickle.dump(clf, f)

            print(f"Treinamento concluído! Features e labels salvos em {PATH_FEATURES} e {PATH_LABELS}")
            clicked_points = []

        if key == 27:
            video.release()
            cv2.destroyAllWindows()
            raise SystemExit

    frame_id += 1

video.release()
cv2.destroyAllWindows()
