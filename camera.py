import warnings
warnings.simplefilter("ignore")

from keras.models import load_model
import cv2
import numpy as np
import SSDFace as ssdf

model = load_model("UTK_age_gender.h5")
ssd = ssdf.cv2_ssd()

def noise():
	test = np.random.randint(256, size =224*224*3)
	test = test.reshape(1,224,224,3)
	return test/255

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

cnt = 0
avgage = 0

while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	facefile = ssd.detect(frame)
	if not facefile is None and len(facefile) > 0:
		cnt+=1

		(x1, y1, x2, y2), conf = facefile[0][0], facefile[0][1]
		if x1<0:x1=0
		if y1<0:y1=0
		face = cv2.resize(frame[y1:y2, x1:x2], (224,224), interpolation=cv2.INTER_CUBIC)
		rst1, rst2 = model.predict(face.reshape(1,224,224,3)/255)

		age = np.argmax(rst1)+11
		gender =  "Male" if np.argmax(rst2) == 0 else "Female"

		avgage+=age

		print("Age:",age, "Gender:",gender, "Avg:",round(avgage/cnt))
	cv2.imshow("Frame", frame)

	#rst1, rst2 = model.predict(noise())
	#print("Age:", np.argmax(rst1)+11, "Gender:", "Male" if np.argmax(rst2) == 0 else "Female")

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
cap.release()
cv2.destroyAllWindows()