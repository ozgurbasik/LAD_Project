
import cv2

# Telefonunuzun IP adresi ve port numarası
ip_address = "http://10.10.210.49:8080/video"

# Kamera akışını başlat
cap = cv2.VideoCapture(ip_address)

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    
    if not ret:
        print("Video akışı alınamadı!")
        break
    
    # Kameradan gelen görüntüyü göster
    cv2.imshow('Phone Camera Feed', frame)

    # 'q' tuşuna basarak çıkışı sağla
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
 