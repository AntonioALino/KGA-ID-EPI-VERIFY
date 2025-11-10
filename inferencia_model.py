import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image
from ultralytics import YOLO



def carregar_arquitetura_resnet():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    return model

inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes_epi = ['without_glasses', 'glasses']


try:
    print("Carregando classificador ResNet18 (epi_model.pth)...")
    resnet_model = carregar_arquitetura_resnet()
    resnet_model.load_state_dict(torch.load('epi_model_v2.pth', map_location=torch.device('cpu')))
    resnet_model.eval()
    print("Classificador ResNet18 carregado.")
except FileNotFoundError:
    print("Erro: 'epi_model.pth' não encontrado.")
    exit()



print("Carregando detector YOLOv8...")

yolo_model = YOLO('yolov8n.pt')
print("Detector YOLOv8 carregado.")


def classificar_crop(crop_img):

    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    
    input_tensor = inference_transforms(pil_image)
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        
        label = classes_epi[preds.item()]
        conf_score = confidence.item()
        
    return label, conf_score

print("Iniciando webcam... Pressione 'q' para sair.")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # 1. DETECÇÃO (YOLO)
    # Roda o YOLO no frame. 'conf=0.4' ignora detecções fracas.
    # 'classes=0' diz ao YOLO para procurar APENAS a classe 'person' (índice 0)
    results = yolo_model(frame, conf=0.4, classes=0, verbose=False)
    
    # 2. CLASSIFICAÇÃO (ResNet18)
    # Iterar sobre todas as caixas (pessoas) que o YOLO encontrou
    for r in results:
        for box in r.boxes:
            # Pega as coordenadas da caixa [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Recortar a pessoa do frame original
            # Adicionamos um 'try' para caso a caixa saia da tela
            try:
                crop = frame[y1:y2, x1:x2]
                
                # Pular recortes muito pequenos (evita erros)
                if crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                # Enviar o recorte para nosso classificador ResNet18
                label_epi, conf_epi = classificar_crop(crop)
                
                # Definir a cor da caixa (Vermelho=sem_epi, Verde=com_epi)
                color = (0, 0, 255) if label_epi == 'sem_epi' else (0, 255, 0)
                
                # Desenhar a caixa do YOLO no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Criar o texto do rótulo
                texto = f"{label_epi.upper()} ({conf_epi*100:.0f}%)"
                
                # Desenhar o texto acima da caixa
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(texto) * 10, y1), color, -1)
                cv2.putText(frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
            except Exception as e:
                print(f"Erro ao processar crop: {e}")
                pass # Ignora erros (ex: caixa vazia)

    # 3. EXIBIÇÃO
    cv2.imshow('Pipeline Detector EPI (YOLO + ResNet18)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Pipeline encerrado.")