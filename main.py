#import mahotas
import pandas as pd
from pylab import gray, imshow, show
import cv2
import os
import numpy as np
import argparse
from skimage import transform
import json

directory = r'D:\University of Surrey\Project\EdgeMatching-master\CuratedCanny'
parent_path = r'D:\University of Surrey\Project\EdgeMatching-master\CuratedSobel_noYOLO'
source_path = r'D:\University of Surrey\Project\EdgeMatching-master\Sobel_noYOLO'

def load_yolo():
	net = cv2.dnn.readNet(r"D:\NeuralFoundry\darknet\cfg\yolov3.weights", r"D:\NeuralFoundry\darknet\cfg\yolov3.cfg")
	classes = []
	with open(r"D:\NeuralFoundry\darknet\data\coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def detect_objects(img, net, outputLayers):
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			# print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    crop = []
    crop1 = []
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            #crop = img[y:y+h, x:x+w]
            #cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    print(crop)
    try:
        crop = img[y:y + h, x:x + w]
    except:
        pass
    if crop == crop1:
        # cv2.imshow("Image", img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        img = cv2.resize(img, (254,254), interpolation = cv2.INTER_AREA)
        x = img
    else:
        try:
            # cv2.imshow("Image", crop)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            crop = cv2.resize(crop, (254, 254), interpolation=cv2.INTER_AREA)
            x = crop

        except:
            pass
    return x
def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    y = draw_labels(boxes, confs, colors, class_ids, classes, image)
	# while True:
	# 	key = cv2.waitKey(100)
    #
		# if key == 27:
		# 	break
    return y


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    r = []
    t = []
    img_blur = []
    subdirs = [x[0] for x in os.walk(directory)]
    for subdir in subdirs:
        #print(os.path.basename(subdir))
        r.append(subdir)
        t.append(os.path.basename(subdir))
        files = os.walk(subdir).__next__()[2]
        # if (len(files) > 0):
        #     for file in files:
        #         #r.append(os.path.join(subdir, file))
    #paths = [f for f in sorted(os.listdir(directory))]
    #print(r[0])
    r.pop(0)
    for path in r:
        print(path)
        current = os.path.join(parent_path,os.path.basename(path))
        os.mkdir(current)
        # print(paths)
        #print(path)
        path1 = os.path.join(source_path,os.path.basename(path))
        for filename in os.listdir(path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image = cv2.imread(os.path.join(path1,filename))
                #image = mahotas.imread(os.path.join(path, filename))
                img = os.path.join(path,filename)
                cv2.imwrite(os.path.join(current, filename), image)
                #print(image)
                # image = image[:,:,0]
                radius = 10
                #print(img)
                #img = cv2.imread("/home/dheeraj/Downloads/dataset/dataset/acinonyx-jubatus/acinonyx-jubatus_61_8d963126.jpg")
                #print(img)
                # Setting parameter values
                #img_blur = cv2.GaussianBlur(img, (3, 3), SigmaX=0, SigmaY=0)     #sobel

                t_lower = 100  # Lower Threshold
                t_upper = 200  # Upper threshold
                aperture_size = 5
                L2Gradient = True

                #z = image_detect(img)
                # Applying the Canny Edge filter
                #try:
                    #image = image[:, :, 0]
                    #value = mahotas.features.zernike_moments(image, radius)
                    # if value[0] == 0:
                    #     continue
                    # with open(os.path.join(current, filename.split(".")[0] + ".json"), "w") as out_file:
                    #     #print(value)
                    #     #value = np.array(value)
                    #     #print(value)
                    #     #points_crop = np.array(points_crop)
                    #     json.dump(value, out_file, cls=NumpyEncoder)
                    # out_file.close()
                    #print(value)
                    #value = np.array(value)
                    #df = pd.DataFrame(value)
                    #print(df)
                    #print(df.shape)
                    #img_blur = cv2.GaussianBlur(z, (3, 3), SigmaX=0, SigmaY=0)
                    #edge = cv2.Sobel(src=z, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                    #edge = cv2.Canny(image, t_lower, t_upper, L2gradient= L2Gradient)
                    # trans_img= transform.rotate(image,angle=50,cval=0)
                    # trans_imgX= transform.rotate(image,angle=-50,cval=0)
                    # trans_imgY= transform.rotate(image, angle=100, cval=0)

                #except:
                    # value = mahotas.features.zernike_moments(image, radius)
                    # if value[0] == 0:
                    #     continue
                    # with open(os.path.join(current, filename.split(".")[0] + ".json"), "w") as out_file:
                    #     #print(value)
                    #     #value = np.array(value)
                    #     #print(value)
                    #     #points_crop = np.array(points_crop)
                    #     json.dump(value, out_file, cls=NumpyEncoder)
                    # out_file.close()
                    #print(value)
                    #pass
                    # image = image[:, :, 0]
                    # value = mahotas.features.zernike_moments(image, radius)
                    # df = pd.DataFrame(value)
                    # print(df.shape)
                    #img_blur = cv2.GaussianBlur(image, (3, 3), SigmaX=0, SigmaY=0)
                    #edge = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                    #edge = cv2.Canny(image, t_lower, t_upper, L2gradient=L2Gradient)
                    # trans_img = transform.rotate(image, angle=50, cval=0)
                    # trans_imgX = transform.rotate(image, angle=-50, cval=0)
                    # trans_imgY = transform.rotate(image, angle=100, cval=0)
                #clone = img

                #image_detect("/home/dheeraj/Downloads/dataset/dataset/acinonyx-jubatus/acinonyx-jubatus_3_fe0c6ea1.jpg")
                # cv2.imshow('original', image)
                # cv2.imshow('edge', edge)
                try:
                    filenameX = "x"+ filename
                    filenameY = "y" + filename
                    filenameZ = "z" + filename
                    filenameA = filename + ".json"
                    # if value[0] == 0:
                    #     continue
                    # with open(os.path.join(current, filename.split(".")[0] + ".json"), "w") as out_file:
                    #     #print(value)
                    #     #value = np.array(value)
                    #     #print(value)
                    #     #points_crop = np.array(points_crop)
                    #     json.dump(value, out_file, cls=NumpyEncoder)
                    # out_file.close()
                    #df.to_json(os.path.join(current,filenameA))
                except:
                    pass
                # cv2.imwrite(os.path.join(current,filename),image)
                # cv2.imwrite(os.path.join(current,filenameX),trans_img)
                # cv2.imwrite(os.path.join(current,filenameY),trans_imgX)
                # cv2.imwrite(os.path.join(current,filenameZ),trans_imgY)
                # print(os.path.join(current,filename))
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
