import pandas as pd
import ast
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os

# Ensure output folder exists
os.makedirs("./output", exist_ok=True)

def denormalize_landmarks(hand_landmarks, image_width, image_height):
    denormalized_landmarks = []
    
    for x, y, z in hand_landmarks:
        x_real = x * image_width
        y_real = y * image_height
        z_real = z * image_width  # Assuming z is scaled relative to width
        denormalized_landmarks.append((x_real, y_real, z_real))
    
    return denormalized_landmarks


cnt = pd.read_csv("./datasets/mudra_vectors.csv")
d = cnt.iloc[:]

for d1 in range(len(d)):  # Iterate over rows properly
    arr = ast.literal_eval(d.iloc[d1]["vector"])  # Parse the stored string list

    arr = np.array(arr)  # Convert to NumPy array

    embedding_vector = arr
    hand_landmarks = []
    
    # Convert flat list to (x, y, z) tuples
    for i in range(0, len(embedding_vector), 3):
        if i + 2 < len(embedding_vector):  # Ensure complete (x, y, z) triplet
            x, y, z = embedding_vector[i], embedding_vector[i + 1], embedding_vector[i + 2]
            hand_landmarks.append((x, y, z))
    
    print(f"Total hand landmarks: {len(hand_landmarks)}")

    # Define image dimensions
    image_width = 256  
    image_height = 256  

    denormalized_hand_landmarks = denormalize_landmarks(hand_landmarks, image_width, image_height)

    # # Find the centroid (average x-value)
    # centroid_x = np.mean([x for x, y, z in denormalized_hand_landmarks])

    # # Rotate the hand 180 degrees to the left (mirror across centroid_x)
    # rotated_hand_landmarks = [(2 * centroid_x - x, y, z) for x, y, z in denormalized_hand_landmarks]

    # denormalized_hand_landmarks = rotated_hand_landmarks

    # OpenCV Visualization
    img_size = 500
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background

    # Extract X, Y for plotting (ignore Z for 2D)
    x_vals = [pt[0] for pt in denormalized_hand_landmarks]
    y_vals = [pt[1] for pt in denormalized_hand_landmarks]
    z_vals = [pt[2] for pt in denormalized_hand_landmarks]

    # Convert Y for OpenCV (flip Y-axis)
    y_vals = [img_size - y for y in y_vals]

    # Draw keypoints on OpenCV image
    for x, y in zip(x_vals, y_vals):
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red points

    # Draw connections (MediaPipe Style)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    connections = mp_hands.HAND_CONNECTIONS  # MediaPipe hand connections

    for p1, p2 in connections:
        if p1 < len(x_vals) and p2 < len(x_vals):
            cv2.line(img, (int(x_vals[p1]), int(y_vals[p1])), 
                          (int(x_vals[p2]), int(y_vals[p2])), (255, 0, 0), 2)  # Blue lines

    # Save the image
    filename = f'./output/hand_skeleton_{d.iloc[d1]["mudra"]}_{d1}.png'
    cv2.imwrite(filename, img)  
    print(f"Saved image: {filename}")

    # # Show image
    # cv2.imshow("Hand Skeleton", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # **Matplotlib 3D Scatter Plot**
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')

    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    # ax.set_title(f'3D Hand Skeleton: {d.iloc[d1]["mudra"]}')

    plt.show()
