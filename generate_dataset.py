import cv2
import os
import time
from ultralytics import YOLO


SAVE_DIR = "Training/dataset_bruto_recortes"
os.makedirs(SAVE_DIR, exist_ok=True)


print("Carregando detector YOLOv8...")
yolo_model = YOLO('yolov8n.pt')
print("Detector YOLOv8 carregado.")


print("Iniciando webcam... Pressione 's' para guardar um frame, 'q' para sair.")
cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        

    results = yolo_model(frame, conf=0.5, classes=0, verbose=False)
    
    frame_desenhado = frame.copy() 
    
    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            

            cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_desenhado, "Pessoa", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            


            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                try:

                    crop = frame[y1:y2, x1:x2]
                    
                    filename = f"crop_{int(time.time() * 1000)}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    cv2.imwrite(filepath, crop)
                    
                    print(f"Guardado: {filepath}")
                    
                except Exception as e:
                    print(f"Erro ao guardar recorte: {e}")
            
            elif key == ord('q'):
                cap.release()
                break
    
    if not cap.isOpened():
        break

    cv2.imshow('Gerador de Dataset - Pressione "s" para guardar', frame_desenhado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Recortes guardados em: {SAVE_DIR}")