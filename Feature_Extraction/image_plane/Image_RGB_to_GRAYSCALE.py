import cv2
import matplotlib.pyplot as plt

folder = ['1', '2', 'z']

for x in range(0, 3):
	file = f"trainALLBlackBoundaryColor/train_o{folder[x]}"

	for y in range(0, 100):
		if(y == 10 or y == 30):
			continue
		img = cv2.imread(f"{file}/open/open.{y}.png")
		#print("Open")
		#print(f"{file}/open/open.{y}.png")

		for i in range(0, 480):
			for j in range(0, 640):
				if(img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255):
					img[i][j][0] = 0
					img[i][j][1] = 0
					img[i][j][2] = 0

				elif(img[i][j][0] == img[i][j][1]):
					img[i][j][0] = 0
					img[i][j][1] = 0
					img[i][j][2] = 0

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (256, 256))

		cv2.imwrite(f"train_o{folder[x]}/open/open.{y}.png", img)

	for y in range(0, 100):
		
		img = cv2.imread(f"{file}/close/close.{y}.png")
	
		for i in range(0, 480):
			for j in range(0, 640):
				if(img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][0] == 255):
					img[i][j][0] = 0
					img[i][j][1] = 0
					img[i][j][2] = 0

				elif(img[i][j][0] == img[i][j][1]):
					img[i][j][0] = 0
					img[i][j][1] = 0
					img[i][j][2] = 0

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (256, 256))

		cv2.imwrite(f"train_o{folder[x]}/close/close.{y}.png", img)
