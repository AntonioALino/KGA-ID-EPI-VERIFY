import cv2
import os
import time
from ultralytics import YOLO


SAVE_DIR = "dataset_bruto_recortes"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. CARREGAR O DETECTOR (YOLOv8) ---
print("Carregando detector YOLOv8...")
yolo_model = YOLO('yolov8n.pt')
print("Detector YOLOv8 carregado.")

# --- 3. LOOP DA WEBCAM ---
print("Iniciando webcam... Pressione 's' para guardar um frame, 'q' para sair.")
cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # --- 4. DETECÇÃO (YOLO) ---
    # Procurar APENAS pessoas (classe 0) com confiança > 0.5
    results = yolo_model(frame, conf=0.5, classes=0, verbose=False)
    
    frame_desenhado = frame.copy() # Copia para desenhar caixas
    
    for r in results:
        for box in r.boxes:
            # Obter coordenadas da caixa
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Desenhar a caixa na cópia do frame
            cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_desenhado, "Pessoa", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # --- 5. LÓGICA PARA GUARDAR (NOVO) ---
            # Verificar se a tecla 's' (de salvar/guardar) foi pressionada
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                try:
                    # Recortar a pessoa do frame ORIGINAL (sem desenhos)
                    crop = frame[y1:y2, x1:x2]
                    
                    # Criar um nome de ficheiro único
                    filename = f"crop_{int(time.time() * 1000)}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    # Guardar o ficheiro de imagem
                    cv2.imwrite(filepath, crop)
                    
                    print(f"Guardado: {filepath}")
                    
                except Exception as e:
                    print(f"Erro ao guardar recorte: {e}")
            
            elif key == ord('q'):
                cap.release()
                break
    
    if not cap.isOpened():
        break

    # Exibir o frame com as caixas (apenas para feedback visual)
    cv2.imshow('Gerador de Dataset - Pressione "s" para guardar', frame_desenhado)
    
    # A lógica de saída 'q' também precisa estar fora do loop for
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Recortes guardados em: {SAVE_DIR}")