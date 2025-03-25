import cv2
import mediapipe as mp
import numpy as np

def main(path):
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Function to calculate angle between three points
    def calculate_angle(a, b, c):
        a = np.array(a)  
        b = np.array(b)  
        c = np.array(c)  

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    # Function to determine palm position
    def get_palm_position(wrist, neck, waist):
        if wrist[1] < neck[1]:  
            return "Near Neck"
        elif wrist[1] > waist[1]:  
            return "Near Waist"
        else:
            return "Center near chest"

    # Load the image
    image_path = path 
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    with mp_pose.Pose(static_image_mode=True) as pose, mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Extract required points
            shoulder_r = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
            shoulder_l = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
            neck = ((shoulder_r[0] + shoulder_l[0]) / 2, (shoulder_r[1] + shoulder_l[1]) / 2)
            waist_r = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
            waist_l = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
            waist = ((waist_r[0] + waist_l[0]) / 2, (waist_r[1] + waist_l[1]) / 2)
            elbow_r = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
            elbow_l = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y)

            # Angle: (Hand, Neck, Shoulder)
            angle_r1 = calculate_angle(shoulder_r, neck, elbow_r)
            angle_l1 = calculate_angle(shoulder_l, neck, elbow_l)

            # Angle: (Shoulder, Waist, Upper Arm)
            angle_r2 = calculate_angle(shoulder_r, waist_r, elbow_r)
            angle_l2 = calculate_angle(shoulder_l, waist_l, elbow_l)

            # Draw Pose Landmarks on Image
            mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
                fingers = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
                elbow = elbow_r if wrist[0] > neck[0] else elbow_l  

                # Angle: (Wrist, Elbow, Fingers)
                angle_hand = calculate_angle(wrist, elbow, fingers)

                # Determine palm position
                palm_position = get_palm_position(wrist, neck, waist)

                # Print Results
                print(f"Hand near {'Right Shoulder' if wrist[0] > neck[0] else 'Left Shoulder'}:")
                print(f"   - Angle (Hand, Neck, Shoulder): {angle_r1 if wrist[0] > neck[0] else angle_l1:.2f}°")
                print(f"   - Angle (Shoulder, Waist, Upper Arm): {angle_r2 if wrist[0] > neck[0] else angle_l2:.2f}°")
                print(f"   - Angle (Wrist, Elbow, Fingers): {angle_hand:.2f}°")
                print(f"   - Palm Position: {palm_position}\n")

                # Draw Hand Landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                )

    # Convert Image Back to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Show Image with Skeleton
    cv2.imshow("Skeleton Image", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main("./assets/test4.jpeg")