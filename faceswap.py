import cv2
import dlib
import numpy as np

# Load dlib's pre-trained face detector and shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect face landmarks
def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected")
    landmarks = predictor(gray, faces[0])
    return landmarks, faces[0]

# Function to swap faces
def swap_faces(image1, image2):
    landmarks1, face1 = get_face_landmarks(image1)
    landmarks2, face2 = get_face_landmarks(image2)

    # Get the facial landmarks
    points1 = np.array([[landmarks1.part(i).x, landmarks1.part(i).y] for i in range(68)])
    points2 = np.array([[landmarks2.part(i).x, landmarks2.part(i).y] for i in range(68)])

    # Create a mask for the faces and extract the face region
    hull1 = cv2.convexHull(points1)
    hull2 = cv2.convexHull(points2)
    
    # Warp the face from image1 to image2 using affine transform
    rect1 = cv2.boundingRect(hull1)
    rect2 = cv2.boundingRect(hull2)

    face1_region = image1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]
    face2_region = image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]]

    # Resize face1 to fit face2
    face1_resized = cv2.resize(face1_region, (face2_region.shape[1], face2_region.shape[0]))

    # Place face1_resized onto image2
    image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = face1_resized

    return image2

# Load images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Perform face swap
result = swap_faces(image1, image2)

# Save the result
cv2.imwrite('swapped_result.jpg', result)
