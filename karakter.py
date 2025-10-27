import cv2
import numpy as np
import os

# KANVAS DAN KARAKTER
canvas = np.full((400, 400, 3), (162, 7, 12), dtype=np.uint8)

# kepala (segitiga)
pts = np.array([[200, 100], [100, 300], [300, 300]], np.int32).reshape((-1,1,2))
cv2.fillPoly(canvas, [pts], (255,255,255))

# mata (dua lingkaran)
cv2.circle(canvas, (160, 170), 20, (0,0,0), -1)
cv2.circle(canvas, (240, 170), 20, (0,0,0), -1)

# mulut (persegi panjang)
cv2.rectangle(canvas, (160, 220), (240, 240), (0,0,255), -1)

# nama karakter
cv2.putText(canvas, "BABY SHARK", (80, 350),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (133,252,240), 2, cv2.LINE_AA)
cv2.imwrite("karakter.png", canvas)

# TRANSFORMASI
# translasi (geser posisi)
dx, dy = 50, 30
M_translate = np.array([[1,0,dx],[0,1,dy]], dtype=np.float32)
translasi = cv2.warpAffine(canvas, M_translate, (canvas.shape[1], canvas.shape[0]))
cv2.imwrite("translasi.png", translasi)

# rotasi (putar)
M_rotate = cv2.getRotationMatrix2D((canvas.shape[1]//2, canvas.shape[0]//2), 30, 1)
rotasi = cv2.warpAffine(translasi, M_rotate, (canvas.shape[1], canvas.shape[0]))
cv2.imwrite("rotasi.png", rotasi)

# resize (ubah ukuran)
resize = cv2.resize(rotasi, None, fx=1.2, fy=1.2)
cv2.imwrite("resize.png", resize)

# crop (potong sebagian gambar dari resize)
h, w = resize.shape[:2]
crop = resize[h//4:3*h//4, w//4:3*w//4]
cv2.imwrite("crop.png", crop)

# OPERASI ARITMATIKA / BITWISE
# bitwise brightness
bitwise_arith = cv2.add(crop, 100)
cv2.imwrite("bitwise_arith.png", bitwise_arith)

# bitwise invert color 
bitwise_negatif = cv2.bitwise_not(bitwise_arith)
cv2.imwrite("bitwise_negatif.png", bitwise_negatif)

# final (hasil semua transformasi)
final = bitwise_negatif
cv2.imwrite("final.png", final)

# tampilkan semua tahap
cv2.imshow("Original", canvas)
cv2.imshow("Translasi", translasi)
cv2.imshow("Rotasi", rotasi)
cv2.imshow("Resize", resize)
cv2.imshow("Crop", crop)
cv2.imshow("Bitwise Arith", bitwise_arith)
cv2.imshow("Bitwise Negatif", bitwise_negatif)
cv2.imshow("Final Character", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
