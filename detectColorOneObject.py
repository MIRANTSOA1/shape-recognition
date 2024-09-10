# Welcome Mirantsoa
# Put your python code for color detection here

import webcolors
from scipy.spatial import KDTree
import colorsys
import cv2
import numpy as np

def get_dominant_color(image_path):

    image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = image_rgb.reshape(-1, 3)

    unique_colors, counts = np.unique(image, axis=0, return_counts=True)
    dominant_color = unique_colors[counts.argmax()]
    
    return tuple(dominant_color)

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

def detect_objects_and_colors(frame):
    """Détecter les objets et leurs couleurs dans une image capturée."""
    # Convertir l'image en espace de couleur HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Définir une plage de couleurs pour le masque (ajuster selon le besoin)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])

    # Créer un masque avec les plages de couleurs
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Trouver les contours des objets détectés par le masque
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ajuster la taille minimale de l'objet
            # Dessiner les contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Calculer le centre du contour pour annoter la couleur
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Extraire la couleur de l'objet détecté au centre du contour
                color_rgb = rgb_frame[cY, cX]
                color_name = get_closest_color_name_hsv(tuple(color_rgb))

                # Afficher le nom de la couleur au centre du contour
                cv2.putText(frame, color_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def main():

    cap = cv2.VideoCapture(0)
    # Définir la résolution de la capture (largeur, hauteur)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Définir les FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Définir le codec

    if not cap.isOpened():
        print("Cam no ouvert")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de lire l'image")
            break
        # Obtenir la couleur dominante de l'image
        dominant_color_rgb = get_dominant_color(frame)
        print(f"Couleur dominante en RGB : {dominant_color_rgb}")

        # Trouver le nom de la couleur la plus proche
        closest_color_name = get_closest_color_name_hsv(dominant_color_rgb)
        print(f"La couleur la plus proche est : {closest_color_name}")

        # Afficher la couleur dominante sur l'image
        cv2.putText(frame, f"Couleur dominante : {closest_color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Afficher l'image avec le texte
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



