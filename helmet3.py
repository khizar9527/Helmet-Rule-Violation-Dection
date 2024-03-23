from ultralytics   import YOLO
import tkinter 
from tkinter import *
from PIL import  Image,ImageTk
import cv2
import cvzone

yolo_helmet = YOLO("Helmet Dataset//Helmet.pt")
yolo_motorcycle = YOLO("motorcyclistdataset//Motorcycle.pt")
yolo_plate = YOLO("platenumber//license_plate_detector.pt")
imgS= None

def plate( x1, y1, x2, y2):
    classname = ['2, 3, 5, 7']
    cropped_image = imgS[y1:y2, x1:x2]
    result = yolo_plate(cropped_image, stream=True)

    for plateornot in result:
        print('1')
        boxes = plateornot.boxes
        for box in boxes:
            print("2")
            c1, d1, c2, d2 = map(int, box.xyxy[0])
            c1, d1 = c1 + x1, d1 + y1
            c2, d2 = c2 + x1, d2 + y1
            cls = int(box.cls[0])
            platename = classname[cls]
            if platename == '2, 3, 5, 7':
                platename='plate'
            cv2.rectangle(imgS, (c1, d1), (c2, d2), (255, 0, 255), 3)
            cvzone.putTextRect(imgS, f'{platename}', (max(0, c1), max(15, d1)), scale=1, thickness=1)
            return

def helmet_detection(x1, y1, x2, y2):
    classname = ['With Helmet', 'Without Helmet']
    cropped_image = imgS[y1:y2,x1:x2]
    result = yolo_helmet(cropped_image, stream=True)

    for helmetornot in result:
        boxes = helmetornot.boxes
        for box in boxes:
            a1, b1, a2, b2 = map(int, box.xyxy[0])
            a1, b1 = a1 + x1, b1 + y1  
            a2, b2 = a2 + x1, b2 + y1
            cls = int(box.cls[0])
            detection1 = classname[cls]
            cv2.rectangle(imgS, (a1, b1), (a2, b2), (255, 0, 255), 3)
            cvzone.putTextRect(imgS, f'{ detection1 }', (max(0, a1), max(35, b2)), scale=1, thickness=0)
            if  detection1 == 'Without Helmet' :
                plate( x1, y1, x2, y2)

            
def motorcycle_detection(image):
    classname = ["bike-riders"]
    result = yolo_motorcycle(image, stream=True)
    detections = []

    for motor in result:
        boxes = motor.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
            cls = int(box.cls[0])
            name = classname[cls]
            detection = {"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            detections.append(detection)

    return detections

def challan():
    global imgS
    cap = cv2.VideoCapture("BB_007f7e21-a0b2-4f50-9c51-db4d6d1f0905_preview.mp4")
    
    frame_skip = 5

    while True:
        for _ in range(frame_skip):
            success,imgS= cap.read()
            if not success:
                break

        if not success:
            break
       # imgS = cv2.resize(img, None, fx=0.25, fy=0.25)
        imgS2 = cv2.resize(imgS, None, fx=0.25, fy=0.25)
        motor_detections = motorcycle_detection(imgS2)

        for motor_detection in motor_detections:
            x1, y1, x2, y2 = motor_detection["x1"], motor_detection["y1"], motor_detection["x2"], motor_detection["y2"]
            name = motor_detection["name"]

            if name == "bike-riders":
                cv2.rectangle(imgS, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(imgS, f'{name}', (max(0, x1), max(15, y1)), scale=1, thickness=1)
                helmet_detection(x1, y1, x2, y2)

        cv2.imshow('Helmet Rule Violation', imgS)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
      

def frame():
    window=Tk()
    window.geometry('1360x728')
    window.title("Helmet Rule Violation Detection")
    canvass=Canvas(window,width=1700,height=900)
    canvass.place(x=0,y=0)
    image=Image.open("upload_image//image5.jpg")
    image2=image.resize((1750,850))
    photo=ImageTk.PhotoImage(image2)
    photo2=canvass.create_image(0,0,image=photo,anchor=NW)
    iconA=Image.open("upload_image//image3.jpg")
    iconA2=ImageTk.PhotoImage(iconA)
    window_button1=Button(window,command=challan,width=164,height=120,image=iconA2) 
    window_button1.pack()
    window_button1.place(x=650,y=310)
    window_button2=Button(window,text="CAPTURE",command=challan,width=18,height=2,bg='Red',fg='white',activebackground='Red',activeforeground='gold',font='time')
    window_button2.pack()
    window_button2.place(x=649,y=436)
    window.mainloop()

frame()