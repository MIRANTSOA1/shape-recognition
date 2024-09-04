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

def main():

    cap = cv2.VideoCapture(0)
    # Définir la résolution de la capture (largeur, hauteur)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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



