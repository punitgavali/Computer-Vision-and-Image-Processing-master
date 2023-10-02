# Overall concept:
# remove the grid lines so that we only have numbers in image 
# find contours in the image which will be numbers
# after finding contours, remove rectangles inside rectangles. 
# For example, digit 8 has three contours, one is big contour of complete digit, and two contours for zeros inside digit 8 so we have to get rid of inside contours 

# Rectangles refer to the bounding boxes of digits
import cv2
import numpy as np

def rectContains(rect1, rect2):
	"""
	returns true if rectangle1 is inside rectangle2
	rect = [startX, startY, Width, Height]
	"""
	return (rect1[0] < rect2[0]+rect2[2] and rect1[0] > rect2[0] and
		rect1[1] < rect2[1]+rect2[3] and rect1[1] > rect2[1])

def removeRectInsideRect(rects):
	"""
	function to remove rectangles which are inside of another rectangle
	"""
	
	insideRects = []
	# loop to find all rectangles which are inside of another rectangle
	for x in range(0, len(rects)):
		for y in range(1, len(rects)):
			if (x != y and rectContains(rects[x], rects[y])):
				insideRects.append(rects[x])

	# loop to extract the rectangles which are not inside of another rectangles
	result = []
	for x in range(0,len(rects)):
		# if rectangle is not one of the inside rectangles, append it to result
		if(rects[x] not in insideRects):
			result.append(rects[x])

	return result
	
def preprocess(img, orig):
	"""
	function to find contours of the digits
	"""

	# color BGR to grayscale conversion
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# applying gaussianblur to smooth image for better thresholding
	gray = cv2.GaussianBlur(img, (5,5), 0)
	# thresholding grayscale image to binary to find better contours
	ret, img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# finding contours inside the binary image
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	# loop to get bounding boxes of the detected contours
	rects = []
	if len(contours)>0:
		for c in contours:
			# checking the area of contour
			if cv2.contourArea(c) > 35 and cv2.contourArea(c)<5000 :
				x,y,w,h = cv2.boundingRect(c)
				rects.append((x,y,w,h))
	# removing rectangles inside of other rectangles
	newRects = removeRectInsideRect(rects)
	# printing red bounding boxes on the image around detected contours aka digits
	for r in newRects:
		cv2.rectangle(orig,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),2)
	# displaying the resultant image
	cv2.imshow('output', orig)
	cv2.waitKey(0)

def removeHorizontalLines(img):
	"""
	Function to remove the horizontal lines from the image
	"""

	# color BGR to grayscale conversion
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# inverting the image
	img = cv2.bitwise_not(img)
	# thresholding the image to perform better morphological operations 
	th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
	horizontal = th2
	# getting the shape of the binary image
	rows,cols = horizontal.shape
	horizontalsize = int(cols / 10)
	# getting structuring element similar to long horizontal line of shape (columnsSize/15, 1)
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
	# eroding the image with above structuring element to remove horizontal line
	horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
	# dilating the image to retain other lost information by eroding. removed horizontal lines won't come back
	horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
	# inverse the image, so that lines are black for masking
	horizontal_inv = cv2.bitwise_not(horizontal)
	# perform bitwise_and to mask the lines with provided mask
	masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
	#reverse the image back to normal
	horizontalRemoved = cv2.bitwise_not(masked_img)
	# convert the grayscale image back to colored (BGR)
	horizontalRemoved = cv2.cvtColor(horizontalRemoved,cv2.COLOR_GRAY2BGR)
	return horizontalRemoved

def removeVerticalLines(img):
	"""
	Function to remove the vertical lines from the image
	"""

	# color BGR to grayscale conversion
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# inverting the image
	img = cv2.bitwise_not(img)
	# thresholding the image to perform better morphological operations 
	th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
	vertical = th2
	# getting the shape of the binary image
	rows,cols = vertical.shape
	verticalsize = int(rows / 12)
	# getting structuring element similar to long vertical line of shape (1, rowSize/10)
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
	# eroding the image with above structuring element to remove vertical line
	vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
	# dilating the image to retain other lost information by eroding. removed vertical lines won't come back
	vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
	#inverse the image, so that lines are black for masking
	vertical_inv = cv2.bitwise_not(vertical)
	#perform bitwise_and to mask the lines with provided mask
	masked_img = cv2.bitwise_and(img, img, mask=vertical_inv)
	#reverse the image back to normal
	verticalRemoved = cv2.bitwise_not(masked_img)
	# convert the grayscale image back to colored (BGR)
	verticalRemoved = cv2.cvtColor(verticalRemoved,cv2.COLOR_GRAY2BGR)
	return verticalRemoved

# reading images from the image files
imgs = []
imgs.append(cv2.imread('sudoku1.jpg'))
imgs.append(cv2.imread('sudoku2.jpg'))
imgs.append(cv2.imread('sudoku3.jpg'))
imgs.append(cv2.imread('sudoku4.jpg'))
imgs.append(cv2.imread('sudoku5.jpg'))

# processing all the loaded images
for x in imgs:
	horizontalRemoved = removeHorizontalLines(x)
	verticalRemoved = removeVerticalLines(horizontalRemoved)
	preprocess(verticalRemoved,x)
