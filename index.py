import os
import pickle
import time
import datetime
import imutils
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUiType
import sys
import face_recognition
from PyQt5.uic.properties import QtWidgets, QtCore, QtGui
from imutils import paths
import cv2
from imutils.video import VideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import sqlite3
from win2 import *
from src_final.win2 import Ui_OtherWindow

conn = sqlite3.connect('mydata.db')
c = conn.cursor()

MainUI, _ = loadUiType('main1.ui')


class Main(MainUI, QMainWindow):
    def ok(self):
        self.windows = QtWidgets.QMainWindow()
        self.ui = Ui_OtherWindow()
        self.ui.setupUi1(self.windows)
        self.windows.show()

    def is_empty_file(self, fpath):
        return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

    def create_tabel(self):
        c.execute("CREATE TABLE IF NOT EXISTS data(name VARCHAR(20), nbr_fois INTEGER , datetime TEXT)")
        conn.commit()

    def dynamic_data_entry(self, mydict):
        self.del_and_update()
        for k in mydict:
            c.execute("""INSERT INTO data(name, nbr_fois, datetime) VALUES (?, ?, ?)""",
                      (k, mydict[k]['nbr_fois'], str(mydict[k]['time'])))
            conn.commit()

    def del_and_update(self):
        c.execute('DELETE FROM  data ')
        conn.commit()

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.Handel_Buttons()
        self.UI_changes()
        self.dir_fichier_encodage = ""
        self.dataset = ""
        self.fichier_encodage = ""
        self.path_input = ""
        self.path_output = ""
        self.fichier_protoxt = "./deploy.prototxt"
        self.fichier_model = "./res10_300x300_ssd_iter_140000.caffemodel"

    def UI_changes(self):
        # Ui changes
        self.tabWidget.tabBar().setVisible(False)



    def Handel_Buttons(self):
        #
        self.pushButton.clicked.connect(self.Open_applicatoin)
        self.pushButton_22.clicked.connect(self.Open_database)
        self.pushButton_3.clicked.connect(self.Open_settings)
        self.btnSelect.clicked.connect(self.onClick)
        self.btn_encodage.clicked.connect(self.encodage)
        self.btn_pickle.clicked.connect(self.pickle_file)
        self.btn_reco.clicked.connect(self.reco_fct)
        self.btnSelect_1.clicked.connect(self.pickle_file)
        self.btnSelect_2.clicked.connect(self.pickle_file)
        self.btn_path_claf.clicked.connect(self.onClick1)
        self.btn_path_output.clicked.connect(self.onClick2)
        self.classification_btn.clicked.connect(self.classification)
        self.btn_detct.clicked.connect(self.detect_file)
        self.btn_track.clicked.connect(self.tracker)
        self.btnshow.clicked.connect(self.ok)
        self.delet_btn.clicked.connect(self.delet_db)


    def delet_db(self):
        msgBox = QMessageBox(self)

        reply = msgBox.question(
            self, 'Attention',
            ("La base de données va totalement supprimer "+
             "Pour désactiver l'action cliquer sur 'no' "),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.del_and_update()
        QMessageBox.information(self, "[INFO]", "Votre base de données est supprimer")

    def reco_image(self, img):
        yes = 0
        no = 0
        names_cle = []
        print("[INFO] loading encodings...")

        data = pickle.loads(open(self.fichier_encodage, "rb").read())
        # image = cv2.imread(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,
                                                model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
        for name in names:
            if name != 'Unknown':
                names_cle.append(name)
                yes += 1
            else:
                # names_cle.append(name)
                # alogrithm for admin the unkouwn names
                # we want to add the face image in the frame
                no += 1

        return yes, no, names_cle

    def tracker(self):
        if self.fichier_encodage == "":
            QMessageBox.warning(self, "[Attention]", "Le fichier d'encodage n'éxite pas")
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.fichier_encodage, _ = QFileDialog.getOpenFileName(self, "Selection le fichier d'encodage", "",
                                                                   "Picke Files (*.pickle)", options=options)
        print("traking is strating")
        self.create_tabel()
        (H, W) = (None, None)
        ct = CentroidTracker(maxDisappeared=50, maxDistance=50)
        trackers = []
        trackableObjects = {}
        fps = FPS().start()
        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        totalFrames = 0
        totalUp = 0
        grab = False
        people_yes = 0
        people_no = 0
        my_dict = {}

        info_value = {"time": [], "nbr_fois": 0}
        data = pickle.loads(open(self.fichier_encodage, "rb").read())

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(self.fichier_protoxt, self.fichier_model)

        if not self.is_empty_file('./infor_time.pickle'):
            for name in data["names"]:
                my_dict[name] = info_value
        else:
            my_dict = pickle.loads(open('./infor_time.pickle', "rb").read())

        # initialize the video stream and allow the camera sensor to warmup
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        j = 0
        
        print("the value of the dictionary is " + str(my_dict))

        while True:
            # read the next frame from the video stream and resize it
            frame = vs.read()
            original = frame.copy()
            # frame = imutils.resize(frame, width=400)
            datetemps = str(datetime.datetime.now().strftime('date : '"%Y/%m/%d  horloge:  %H:%M:%S"))
            cv2.putText(frame, datetemps, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (213, 255, 0), 1, cv2.LINE_AA)

            # if the frame dimensions are None, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # construct a blob from the frame, pass it through the network,
            # obtain our output predictions, and initialize the list of
            # bounding box rectangles
            blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                         (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            rects = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # filter out weak detections by ensuring the predicted
                # probability is greater than a minimum threshold
                if detections[0, 0, i, 2] > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object, then update the bounding box rectangles list
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    rects.append(box.astype("int"))

                    # draw a bounding box surrounding the object so we can
                    # visualize it
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

            # update our centroid tracker using the computed set of bounding
            # box rectangles
            objects = ct.update(rects)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True
                            if not grab:
                                #cv2.imwrite("output/frme%d.jpg" % j, original)
                                j += 1
                                # print("j = "+ str(j))
                                # here we have to add the image into our model recognition for get the resuslt

                                yes, no, names_cle = self.reco_image(original)

                                people_yes += yes
                                people_no += no
                                print("the dictoinary before the line " + str(my_dict))
                                for name in names_cle:
                                    if my_dict[name] == info_value:
                                        my_dict[name] = {
                                            'time': [str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
                                            'nbr_fois': 1}
                                    # my_dict["salah_eddine"]["time"].append(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                    else:
                                        info_value_1 = my_dict[name]
                                        info_value_1["nbr_fois"] += 1
                                        info_value_1["time"].append(
                                            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                # self.dynamic_data_entry(my_dict)
                f = open("./infor_time.pickle", "wb")
                f.write(pickle.dumps(my_dict))
                f.close()

                trackableObjects[objectID] = to
                text = "ID {}".format(objectID)
                text1 = "les connues : " + str(people_yes)
                cv2.putText(frame, text1, (5, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23, 108, 71), 2)

                '''text2 = "les inconnues : " + str(people_no)
                cv2.putText(frame, text2, (5, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (243, 73, 73), 1)'''

                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                text1 = "Depassement la ligne : " + str(totalUp)
                cv2.putText(frame, text1, (10, H - ((0 * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
            # show the output frame
            cv2.imshow("compter et suivre les personnes", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            totalFrames += 1
            fps.update()

        # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        vs.stop()
        cv2.destroyAllWindows()
        print("la fin de l'application")
        self.dynamic_data_entry(my_dict)

    def detect_file(self):
        if (os.path.isfile(self.fichier_protoxt) and self.fichier_protoxt.endswith(".prototxt")) and os.path.isfile(
                self.fichier_model) and self.fichier_model.endswith(".caffemodel"):
            QMessageBox.information(self, "[INFO]", "les fichier de model de détection est bien détecter")
        else:
            QMessageBox.warning(self, "Attention", "les fichiers de model n'exist pas !")



    def classification(self):
        if self.fichier_encodage == "":
            QMessageBox.warning(self, "[Attention]", "Le fichier d'encodage n'éxite pas")
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.fichier_encodage, _ = QFileDialog.getOpenFileName(self, "Selection le fichier d'encodage", "",
                                                                   "Picke Files (*.pickle)", options=options)
        if self.fichier_encodage == '' or self.path_output == '' or self.path_input =='':
            QMessageBox.warning(self, "Attention", "vérifier l'existence des fichier et des dossier")
        else:
            print("[INFO] loading encodings...")
            QMessageBox.information(self, "[INFO]", " charger l'encodage ...")
            data = pickle.loads(open(self.fichier_encodage, "rb").read())
            for root, dirs, files in os.walk(self.path_input):
                for file in files:
                    if file.endswith("png") or file.endswith('jpg'):
                        print(os.path.join(root, file))
                        image = cv2.imread(os.path.join(root, file))
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        print("[INFO] recognizing faces...")
                        QMessageBox.information(self, '[INFO', "Reconnaissance les visages dans l'image...")

                        cv2.imshow("Image", image)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                        boxes = face_recognition.face_locations(rgb,
                                                                model='hog')
                        encodings = face_recognition.face_encodings(rgb, boxes)
                        names = []

                        # loop over the facial embeddings
                        for encoding in encodings:
                            # attempt to match each face in the input image to our known
                            # encodings
                            matches = face_recognition.compare_faces(data["encodings"],
                                                                     encoding)
                            name = "Inconnu"

                            # check to see if we have found a match
                            if True in matches:
                                # find the indexes of all matched faces then initialize a
                                # dictionary to count the total number of times each face
                                # was matched
                                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                                counts = {}
                                # loop over the matched indexes and maintain a count for
                                # each recognized face face
                                for i in matchedIdxs:
                                    name = data["names"][i]
                                    counts[name] = counts.get(name, 0) + 1

                                # determine the recognized face with the largest number of
                                # votes (note: in the event of an unlikely tie Python will
                                # select first entry in the dictionary)
                                name = max(counts, key=counts.get)

                            # update the list of names
                            names.append(name)
                        i = 0
                        for name in names:
                            if name == 'Inconnu':
                                pass
                            else:
                                path = self.path_output + '/' + str(name)
                                isdir = os.path.isdir(path)
                                if not isdir:
                                    try:
                                        os.mkdir(str(path))
                                    except OSError:
                                        print("Creation of the directory /s filed" % path)
                                    else:
                                        pass
                                    # print("Succely created the directory "% path)
                                print("i = " + str(i))
                                print("the path is " + str(path))
                                cv2.imwrite(str(path) + '/' + str(name) + str(i) + '.jpg', image)
                                i += 1

            QMessageBox.information(self, "INFO", "Classification est terminé....")

    def onClick2(self):
        self.path_output = str(
            QFileDialog.getExistingDirectory(self, "Select le Dossier pour Enregister les resultat final"))
        print("path ot resultat " + self.path_output)

    def onClick1(self):
        self.path_input = str(QFileDialog.getExistingDirectory(self, "Select votre dossier de images"))
        print("path input  " + self.path_input)
        for dirpath, dirnames, files in os.walk(self.path_input):
            if dirnames is not [] and files:
                print("the folder is not empty")
                break
            elif dirnames == []:
                QMessageBox.warning(self, "Attention", "Le Dessier que vous sélectionnez est vide")

    def pickle_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fichier_encodage, _ = QFileDialog.getOpenFileName(self, "Selection le fichier d'encodage", "",
                                                               "Picke Files (*.pickle)", options=options)

    def Open_applicatoin(self):
        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)

    def Open_database(self):
        self.tabWidget.setCurrentIndex(1)

    def reco_fct(self):
        if self.fichier_encodage == "":
            QMessageBox.warning(self, "[Attention]", "Le fichier d'encodage n'éxite pas")
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.fichier_encodage, _ = QFileDialog.getOpenFileName(self, "Selection le fichier d'encodage", "",
                                                                   "Picke Files (*.pickle)", options=options)

        print("[INFO] loading encodings...")
        QMessageBox.information(self, "[INFO]", "Charger le fichier d'encodage")
        data = pickle.loads(open(self.fichier_encodage, "rb").read())
        print("[INFO] starting video stream...")
        QMessageBox.information(self, "[INFO]", "WebCam est start")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        # print("Reconnaissance mode = " +self.mode_reconnissance)
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb,
                                                    model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # loop over the facial embeddings
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                name = "Inconnu"

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 150, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 0, 0), 2)

            cv2.imshow("Reconnaissance Facial", frame)
            key = cv2.waitKey(1) & 0xFF

            # si le button q est press la boucle est fini
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def Open_settings(self):
        self.tabWidget.setCurrentIndex(2)

    def onClick(self):
        self.dataset = str(QFileDialog.getExistingDirectory(self, "Sélectionner DataSet"))
        for dirpath, dirnames, files in os.walk(self.dataset):
            if dirnames is not [] and files:
                break
            elif dirnames == []:
                QMessageBox.warning(self, "Attention", "DataSet est vide")

        print("the dir is " + str(self.dataset))
        # paht = os.path.realpath(os.path.,self.fichier_encodage)
        # print("the relative deirectory is " + str(paht))

    def encodage(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("[INFO]")
        knownEncodings = []
        knownNames = []
        # text = ""
        text = "[INFO] Quantification des visages..." + '\n'
        text += "[INFO] cette étape est trés long s'il vous attendez un peu de temps...." + '\n'
        print("[INFO] Quantification des visages...")
        # QMessageBox.information(self, "[INFO]", "Quantification des visages...")
        ficher_encodage = self.in_encodage.text()
        if len(ficher_encodage) == 0:
            QMessageBox.warning(self, "Attention", "Enter le nom de fichier d'encodage")
            ficher_encodage = self.in_encodage.text()
            if len(ficher_encodage) > 0:
                QMessageBox.information(self, "[INFO]", "Quantification des visages...")

        ficher_encodage += '.pickle'
        self.dir_fichier_encodage += './' + ficher_encodage
        self.label_3_.setText(text)
        imagePaths = list(paths.list_images('./dataset'))
        QMessageBox.information(self, "[INFO]",
                                "Cette Etat prendre un peu de temps s'il vous plait attent un peu de temps")
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            text += 'ici  la photo est '+ str(imagePath)
            text += "[INFO] traitement de l'image {}/{}".format(i + 1,
                                                                len(imagePaths)) + '\n'
            print(text)
            self.label_3_.setText(text)
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from RGB (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, model='hog')

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and encodings
                knownEncodings.append(encoding)
                knownNames.append(name)
        # dump the facial encodings + names to disk
        text += "[INFO] sérialisation des encodages ..." + '\n'
        print(text)
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open(self.dir_fichier_encodage, "wb")
        f.write(pickle.dumps(data))
        f.close()
        text += "[INFO] encodage est terminé ..." + '\n'
        self.label_3_.setText(text)
        QMessageBox.information(self, "[INFO]", "[INFO] encodage est terminé ...")


def main():
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
