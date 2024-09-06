import cv2
import torch
import segmentation_models_pytorch as smp
import numpy as np
import time

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=2)
model.load_state_dict(torch.load("/run/media/berkkucukk/sandisk/PythonProjects/AutoPilot/best_model.pth"))
model.eval()

input_size = (256, 256)

video_path = "/run/media/berkkucukk/sandisk/PythonProjects/AutoPilot/test1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video dosyası bulunamadı veya açılamadı.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = "/run/media/berkkucukk/sandisk/PythonProjects/AutoPilot/predicted_video.mp4"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    input_image_resized = cv2.resize(input_image, input_size)
    
    input_tensor = torch.tensor(input_image_resized.transpose(2, 0, 1), dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  
    
    with torch.no_grad():
        output = model(input_tensor)
    
    mask = (output.argmax(1).squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    mask_resized = cv2.resize(mask, frame.shape[:2][::-1])
    
    mask_colored = np.zeros_like(frame)
    mask_colored[:, :, 1] = mask_resized  

    masked_image = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)
    
    out.write(masked_image)
    
    cv2.imshow('Lane Detection', masked_image)
    cv2.waitKey(100)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("İşlenmiş video başarıyla kaydedildi:", output_video_path)
