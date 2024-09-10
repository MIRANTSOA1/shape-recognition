import cv2
import imutils
import numpy as np
import webcolors
from scipy.spatial import KDTree
import colorsys

def rgb_to_hsv(rgb_color):
    r, g, b = rgb_color
    hsv = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hsv_scaled = (hsv[0] * 360, hsv[1] * 100, hsv[2] * 100)
    return hsv_scaled

def get_closest_color_name_hsv(requested_rgb):
    requested_hsv = rgb_to_hsv(requested_rgb)

    css3_colors = webcolors.names("css3")

    css3_colors_hex = {}

    for color_name in css3_colors:
        try:
            rgb_color = webcolors.name_to_rgb(color_name)
            hex_color = webcolors.rgb_to_hex(rgb_color)
            css3_colors_hex[color_name] = hex_color
        except ValueError:
            pass
    color_names = list(css3_colors_hex.keys())    
    color_rgb_values = [webcolors.hex_to_rgb(hex_value) for hex_value in css3_colors_hex.values()]
    color_hsv_values = [rgb_to_hsv(rgb) for rgb in color_rgb_values]

    kdt_db = KDTree(color_hsv_values)
    distance, index = kdt_db.query(requested_hsv)
    closest_color_name = color_names[index]

    return closest_color_name

def detect_shape(contour):
    # Calculer le périmètre du contour
    peri = cv2.arcLength(contour, True)
    # Approximation des contours
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Identifier la forme selon le nombre de sommets
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        # Calculer le rapport largeur/hauteur pour déterminer s'il s'agit d'un carré ou d'un rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Carré" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif len(approx) == 5:
        return "Pentagone"
      
    elif len(approx) == 10 or len(approx) == 12:
        return "Etoile"
    else:
        return "Cercle"

def main():
    # Initialiser la capture vidéo
    cap = cv2.VideoCapture(0)

    # Vérifier si la caméra s'ouvre correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la caméra.")
        return

    while True:
        # Lire une image de la caméra
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image.")
            break

        # Redimensionner l'image pour une meilleure performance
        resized = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hsv_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Appliquer le filtre gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Utiliser le filtre Canny pour détecter les bords
        edged = cv2.Canny(blurred, 50, 150)

        # Utiliser un seuillage adaptatif pour obtenir une image binaire
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Trouver les contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Parcourir chaque contour détecté
        for contour in contours:
            # Calculer l'aire et filtrer les petits contours
            area = cv2.contourArea(contour)
            if area < 100:  # Seuil de filtrage des petits contours
                continue

            # Calculer le centre de la forme
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Détecter la forme
            shape_name = detect_shape(contour)
            color_rgb = rgb_frame[cY, cX]
            color_name = get_closest_color_name_hsv(tuple(color_rgb))
            # Dessiner le contour et le nom de la forme
            cv2.drawContours(resized, [contour], -1, (55, 183, 195), 2)
            cv2.putText(resized, f"{shape_name}, {color_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Afficher l'image avec les formes détectées
        cv2.imshow('Shapes Detection', resized)

        # Quitter en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la caméra et fermer toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
