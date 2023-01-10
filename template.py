import os
import cv2
import numpy as np

def get_path_list(root_path):
    train_names = os.listdir(root_path)
    return train_names
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

def get_class_id(root_path, train_names):
    image_classes_list = []
    train_image_list = []
    for i, name in enumerate(train_names):
        class_path_list = os.listdir(root_path + '/' + name)
        for image_path in class_path_list:
            image_classes_list.append(i)
            train_image_path = root_path + '/' + name + '/' + image_path
            train_image = cv2.imread(train_image_path)
            train_image_list.append(train_image)
    return train_image_list, image_classes_list

    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''

def detect_faces_and_filter(image_list, image_classes_list=None):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_list = []
    class_list = []
    rectangle_list = []

    if(image_classes_list==None):
        for image in image_list:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5)
            if(len(detected_faces) < 1):
                continue
            for rectangle in detected_faces:
                x, y, w, h = rectangle
                face_image = image_gray[y:y+w, x:x+h]
                face_list.append(face_image)
                rectangle_list.append(rectangle)
    
    else:
        for image, class_id in zip(image_list, image_classes_list):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5)
            if(len(detected_faces) < 1):
                continue
            for rectangle in detected_faces:
                x, y, w, h = rectangle
                face_image = image_gray[y:y+w, x:x+h]
                face_list.append(face_image)
                rectangle_list.append(rectangle)
                class_list.append(class_id)            
    
    return face_list,rectangle_list,class_list



    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path):
    test_image_list = []
    test_path_list = os.listdir(test_root_path)

    for image_path in test_path_list:
        test_image_path = test_root_path + '/' + image_path
        test_image_gray = cv2.imread(test_image_path)
        test_image_list.append(test_image_gray)
        
    return test_image_list
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    
def predict(recognizer, test_faces_gray):
    prediction_list = []
    for image in test_faces_gray:
        result, _ = recognizer.predict(image)
        prediction_list.append(result)
    
    return prediction_list
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    draw_image_list = []
    for result, ractangle, image in zip(predict_results,test_faces_rects,test_image_list):
        x, y, w, h = ractangle
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 5)
        resize_image = cv2.resize(image, (350,350))
        text = train_names[result]
        cv2.putText(resize_image, text, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
        draw_image_list.append(resize_image)
    
    return draw_image_list
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    

def combine_and_show_result(image_list):

    cv2.imshow("final result", cv2.hconcat(image_list))
    cv2.waitKey(0)
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)

    combine_and_show_result(predicted_test_image_list)
    

