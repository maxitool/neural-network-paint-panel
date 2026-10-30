from tkinter import *
from colorutils import Color
import ctypes
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from PIL import Image
import sqlite3

def Read_Data_From_File(fileName):
    data = []
    with open(fileName + ".txt") as f:
        for line in f:            
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    return data
def Draw_Image_From_Array(data):
    imgData = data.copy()
    imgData = imgData * 255
    for i in range(len(imgData)):
        for j in range(len(imgData[i])):
            imgData[i][j] = 0 - (imgData[i][j] - 255)
    imgData = np.array(imgData, dtype=np.uint8)
    img = Image.fromarray(imgData, 'L')
    img.save('Image.png')
def Draw(event):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval((x1, y1, x2, y2), fill='black', width=0)
def Clear_Canvas():
    canvas.delete('all')
    canvas['bg'] = 'white'
def Predict_Hieroglyph():
    data = Get_Pixels_Of_Canvas()
    Write_Data_In_File(data)
    err = Normalization_From_Cpp()
    if err != 0:
        return
    data = Read_Data_From_File('data')
    Draw_Image_From_Array(data)
    Neural_Network_Work(data)

def Neural_Network_Work(data):
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=3)
    result1 = Neural_Network_Predict(-1, 0.2, data)
    numberClass = 0
    if result1[0][0] != 53:
        for i in range(len(result1)):
            result2 = Neural_Network_Predict(result1[i][0], 0.55, data)
            if len(result2) > 0:
                numberClass = i
                break
    if result1[0][0] == 53:
        image = Image.open('Images/i512.png')
        image.show()
        return
    if len(result2) > 0:
        result = Output_Kanji(result1[numberClass][0], result2[0][0])
        idKanji = Get_ID_FROM_SQLITE(result)
        image = Image.open('Images/i' + str(idKanji) + '.png')
        image.show()
    a = 2

def Neural_Network_Predict(classNumber, minValue, data):
    countClasses = Count_Of_Classes(classNumber)
    model = Model_Create(countClasses)
    if classNumber == -1:
        model.load_weights('Weights/Main.h5')
    else:
        model.load_weights('Weights/' + str(classNumber) + '.h5')
    result = model.predict(data)
    result = Get_Results_NN(minValue, result)
    return  result

def Output_Kanji(classNumber, kanjiNumber):
    iClass = 0
    iKanji = 0
    kanji = ''
    currentClass = 0
    with open("ALL_CLASS_KANJI.txt", encoding="utf8") as f:
        for line in f:
            if currentClass == classNumber:                
                kanji = line[len(str(classNumber)) + kanjiNumber]
                break
            currentClass = currentClass + 1
    return kanji

def Count_Of_Classes(number):
    i = 0
    countClasses = -1
    with open("CountClasses.txt") as f:
        for line in f:
            if i == number + 1:
                 countClasses = int(line)
                 break
            i = i + 1
    return countClasses
def Get_Results_NN(key, result):
    data = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j] >= key:
                data.append([j, result[i][j]])
    data = MySort(data)
    return data

def MySort(data):
    result = []
    max = 0
    countClasses = len(data)
    numberData = 0
    while countClasses > 0:
        for i in range(len(data)):
            if data[i][1] > max:
                max = data[i][1]
                numberData = i
        result.append(data[numberData].copy())
        data[numberData][1] = 0
        max = 0
        countClasses = countClasses - 1
    return result

def Normalization_From_Cpp():
    file = ctypes.CDLL('./NormalizationEx3.so')
    file.NormalizationCpp.restype = ctypes.c_int
    err = file.NormalizationCpp()
    return err
def Write_Data_In_File(data):
    with open('data.txt','w') as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                f.write(str(data[i][j]) + ' ')
            f.write('\n')
def Get_Pixels_Of_Canvas():
    width = int(canvas["width"])
    height = int(canvas["height"])
    colors = []
    for x in range(width):
        column = []
        for y in range(height):
            column.append(get_pixel_color(x, y))
        colors.append(column)
    return colors
def get_pixel_color(x, y):
    ids = canvas.find_overlapping(y, x, y, x)
    if len(ids) > 0:
        index = ids[-1]
        color = canvas.itemcget(index, "fill")
        if color == 'black':
            return 1.0
    return 0.0
def Model_Create(countClasses):
    model = keras.Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,1)),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=2),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(countClasses, activation='softmax')
        ])
    return model
def Get_ID_FROM_SQLITE(kanji):
    cursor.execute("SELECT ID FROM INFO WHERE KANJI = ?", (kanji,))
    idKanji = cursor.fetchone()
    return idKanji[0]

conn = sqlite3.connect('kanji_information.db')
cursor = conn.cursor()
root = Tk()
root.title('NN_Panel_Paint')
root.geometry('640x640')
root.resizable(0, 0)
root.columnconfigure(2, weight=1)
root.rowconfigure(2, weight=1)
brush_size = 12
canvas = Canvas(root, bg='white', width=640, height=640)
canvas.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky=E+W+S+N)
canvas.bind('<B1-Motion>', Draw)
canvas.bind('<Button-1>', Draw)
menu = Menu(tearoff=0)
Button(root, text='Очистить', width=10, command=Clear_Canvas).grid(row=0, column=0, padx=5)
Button(root, text='Предсказать', width=10, command=Predict_Hieroglyph).grid(row=0, column=1, padx=5)
root.mainloop()
