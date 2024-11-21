import cv2
import numpy as np

# Define globals
MAX_SIZE = 406

# Define helper functions
def extractAruco(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Inverse threshold to get the inner contour
    ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Aucun contour trouvé")
        return None
    
    # Crop image using bounding rectangle
    x, y, w, h = cv2.boundingRect(contours[0])
    image_crop = gray[y:y + h, x:x + w]
    cv2.imshow("Crop", image_crop)
    
    return image_crop

def findArucoID(inp_img):
    # Extract image
    im = extractAruco(inp_img)
    if im is None:
        return None  # Si l'image extraite est vide, retournez None

    # Resize it to a smaller size for check
    im = cv2.resize(im, (MAX_SIZE, MAX_SIZE))

    # Remove padding
    width = int(MAX_SIZE / 7)
    im = im[width:width * 6, width:width * 6]
    cv2.imshow("Without Padding", im)

    # Calculate ID
    ret_val = 0
    bits_matrix = []  # Matrice pour stocker les bits

    for y in range(5):
        # Read bits from the resized image
        _val1 = int(im[int((y * width) + (width / 2)), int(width + width / 2)])
        _val2 = int(im[int((y * width) + (width / 2)), int(3 * width + width / 2)])

        # Convert pixel values to binary
        _val1 = 1 if _val1 == 255 else 0
        _val2 = 1 if _val2 == 255 else 0

        ret_val = ret_val * 2 + _val1
        ret_val = ret_val * 2 + _val2

        # Store bits in matrix
        bits_matrix.append([_val1, _val2])

    # Print the bits matrix
    print("Bits matrix:")
    for row in bits_matrix:
        print(row)

    return ret_val

# Load the image
image = cv2.imread("aruco_fixed.png")

if image is None:
    print("Erreur : Impossible de lire l'image. Vérifiez le chemin et le fichier.")
    exit()

# Find the ArUco ID
ID = findArucoID(image)

if ID is not None:
    print("Marker ID is", ID)
else:
    print("Marker ID could not be determined.")

cv2.waitKey(0)
cv2.destroyAllWindows()
