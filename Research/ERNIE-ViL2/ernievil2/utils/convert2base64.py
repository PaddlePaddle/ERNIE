import cv2
import base64
import numpy as np
def image_file_to_str(filename):
    img = cv2.imread(filename)
    img_encode = cv2.imencode(".jpg", img)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    return base64.b64encode(str_encode)
if __name__ == "__main__":
    print (image_file_to_str)