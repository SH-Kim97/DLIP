import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sudokuSolver import *


def detectSudoku(pathImage):
    heightImg = 450
    widthImg = 450
    model = intializePredectionModel() # Load the CNN model

    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))
    imgThreshold = preProcess(img)

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest, maxArea = biggestContour(contours) # Find the biggest contour

    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgDetectedDigits = np.ones((heightImg, widthImg, 3), np.uint8) * 255

        # Split the image and find each digit available
        boxes = splitBoxes(imgWarpColored)
        numbers = getPredection(boxes, model)
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers)
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)
        inNumbers = np.where(numbers > 0, -1, 0)

        # Find solution of the board
        board = np.array_split(numbers, 9)
        try:
            solve(board)
        except:
            pass

        # Save the solution
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers = flatList*posArray
        solvedNumbers = np.where(solvedNumbers == 0, -1, solvedNumbers)

        # Draw grid and Return
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        return numbers.tolist(), inNumbers.tolist(), solvedNumbers.tolist(), imgDetectedDigits

    else:
        print("No Sudoku Found")

# Draw selected grid
def selectedGrid(img, pos):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)

    xpos, ypos = pos % 9, pos // 9
    pt1 = (xpos*secW+3, ypos*secH+3)
    pt2 = ((xpos+1)*secW-3, (ypos+1)*secH-3)

    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    return img

# Display input numbers
def inNumbers(img, numbers, color = (255, 0, 0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] not in [-1, 0] :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-12, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img