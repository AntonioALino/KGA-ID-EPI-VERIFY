import os
import shutil
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import glob

class ImageOrganizer:
    def __init__(self, source_folder, target_folders):
        self.source_folder = source_folder
        self.target_folders = target_folders
        
        # Criar pastas destino se não existirem
        for folder in target_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Listar todas as imagens
        self.images = glob.glob(os.path.join(source_folder, "*.jpg"))
        self.images.extend(glob.glob(os.path.join(source_folder, "*.png")))
        self.current_index = 0
        
        # Configurar interface
        self.root = tk.Tk()
        self.root.title("Organizador de Imagens para Treinamento")
        
        # Exibir imagem
        self.img_label = ttk.Label(self.root)
        self.img_label.pack()
        
        # Botões
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        for label, folder in target_folders.items():
            ttk.Button(button_frame, text=label, 
                      command=lambda f=folder: self.move_to_folder(f)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Pular", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Mostrar primeira imagem
        self.show_current_image()
        
    def show_current_image(self):
        if self.current_index < len(self.images):
            img_path = self.images[self.current_index]
            img = Image.open(img_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            
            self.img_label.config(image=photo)
            self.img_label.image = photo
            
            self.root.title(f"Imagem {self.current_index + 1} de {len(self.images)}")
    
    def move_to_folder(self, target_folder):
        if self.current_index < len(self.images):
            img_path = self.images[self.current_index]
            filename = os.path.basename(img_path)
            destination = os.path.join(target_folder, filename)
            
            shutil.move(img_path, destination)
            print(f"Movido: {filename} -> {target_folder}")
            
            self.next_image()
    
    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.images):
            self.show_current_image()
        else:
            print("Todas as imagens foram processadas!")
            self.root.quit()
    
    def run(self):
        self.root.mainloop()

# Uso:
if __name__ == "__main__":
    # Defina os caminhos conforme sua estrutura
    source = "Training/dataset_bruto_recortes"
    targets = {
        "COM EPI": "Training/com_epi",
        "SEM EPI": "Training/sem_epi"
    }
    
    organizer = ImageOrganizer(source, targets)
    organizer.run()