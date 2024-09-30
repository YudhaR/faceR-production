import cv2

# Buka kamera
cap = cv2.VideoCapture(0)  # 0 untuk kamera default

# Loop untuk menangkap frame dari kamera
while cap.isOpened():
    # Membaca frame dari kamera
    success, frame = cap.read()

    if success:
        # Tampilkan frame asli dari kamera
        cv2.imshow("Kamera OpenCV", frame)

        # Break the loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop jika stream kamera tidak tersedia
        break

# Lepaskan objek video capture dan tutup jendela tampilan
cap.release()
cv2.destroyAllWindows()
