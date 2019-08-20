import numpy as np, cv2#, matplotlib.pyplot as plt

class cv2_ssd:
	def __init__(self, threshold=0.5):
		self.t = threshold
		self.prototxt = 'deploy.prototxt.txt'
		self.caffemodel = 'res10_300x300_ssd_iter_140000.caffemodel'
		self.detector = self._create_ssd_detector()

	def _create_ssd_detector(self):
		ssd = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)

		return ssd

	def _dnn_blob(self, image):
		img = cv2.resize(image, (300,300), interpolation = cv2.INTER_CUBIC)
		blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

		return blob

	def _ssd_box(self, detection, h, w):
		rects = []

		for i in range(0,detection.shape[2]):
			confidence = detection[0, 0, i, 2]

			if confidence > self.t:
				box = detection[0, 0, i, 3:8] * np.array([w, h, w, h])
				(x1, y1, x2, y2) = box.astype('int')
				rects.append(((x1, y1, x2, y2), confidence))

		return rects

	def detect(self, img):
		h, w = img.shape[:2]
		blob = self._dnn_blob(img)
		
		self.detector.setInput(blob)
		detections = self.detector.forward()

		faces = self._ssd_box(detections, h, w)
		return faces

#if __name__ == '__main__':
#	threshold = 0.5

	#img = cv2.imread('imgs/bike.jpg')

#	ssd = cv2_ssd(threshold = threshold)
#	faces = ssd.detect(img)

#	pic = img.copy()
#	for i in range(len(faces)):
#		x1, y1 = faces[i][0][:2]
#		x2, y2 = faces[i][0][2:]
#		confidence = faces[i][1]
#		cv2.rectangle(pic, (x1, y1), (x2, y2), (0,0,255), 3)

		#y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#		cv2.putText(pic, str(confidence), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

#	plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
#	plt.show()