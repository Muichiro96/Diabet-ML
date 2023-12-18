import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
from model import Predictor

window = tk.Tk()
window.title("Diabetes analysis")
photo = tk.PhotoImage(file="diabetes-test.png")
window.wm_iconphoto(False, photo)
background_image = tk.PhotoImage(file="bg4.png")
frame = tk.Frame(master=window, width=600, height=600)
frame.pack()
background_label = tk.Label(frame, image=background_image)
background_label.place(relwidth=1, relheight=1)
predictor = Predictor()


def eventhandler(e):
    print(pregnancy.get(), glucoselevel.get(), bloodPressure.get(), skinThickness.get(), insulinLevel.get(), BMI.get(),
          DPF.get(), age.get())
    result = predictor.isDiabetic(getInputData())
    if result:
        res.config(text="results", font="Arial")
        res1.config(text="You are Diabetic", bg="red",
                    font="Helvetica 16 bold italic")
    else:
        res.config(text="results", font="Arial")
        res1.config(text="You are not Diabetic", fg="light green", bg="dark green",
                    font="Helvetica 16 bold italic")
    # TODO check if true or false and show results to user!


def getInputData():
    return [pregnancy.get(), glucoselevel.get(), bloodPressure.get(), skinThickness.get(), insulinLevel.get(),
            BMI.get(),
            DPF.get(), age.get()]


label1 = tk.Label(master=frame, text="Blood Pressure :")
label1.place(x=30, y=50)
label1.config(font=("Courier New", 12))
bloodPressure = tk.Entry(master=frame)
bloodPressure.place(x=320, y=50)
label2 = tk.Label(master=frame, text="Number Pregnacy :")
label2.place(x=30, y=100)
label2.config(font=("Courier New", 12))
pregnancy = tk.Entry(master=frame)
pregnancy.place(x=320, y=100)
label21 = tk.Label(master=frame, text="Insulin Level :")
label21.place(x=30, y=150)
label21.config(font=("Courier New", 12))
insulinLevel = tk.Entry(master=frame)
insulinLevel.place(x=320, y=150)
label3 = tk.Label(master=frame, text="Glucose Level :")
label3.place(x=30, y=200)
label3.config(font=("Courier New", 12))
glucoselevel = tk.Entry(master=frame)
glucoselevel.place(x=320, y=200)
label4 = tk.Label(master=frame, text="Skin Thickness :")
label4.place(x=30, y=250)
label4.config(font=("Courier New", 12))
skinThickness = tk.Entry(master=frame)
skinThickness.place(x=320, y=250)
label5 = tk.Label(master=frame, text="BodyMassIndex :")
label5.place(x=30, y=300)
label5.config(font=("Courier New", 12))
BMI = tk.Entry(master=frame)
BMI.place(x=320, y=300)
label6 = tk.Label(master=frame, text="Diabetes Pedigree Function :")
label6.place(x=30, y=350)
label6.config(font=("Courier New", 12))
DPF = tk.Entry(master=frame)
DPF.place(x=320, y=350)
label7 = tk.Label(master=frame, text="Age :")
label7.place(x=30, y=400)
label7.config(font=("Courier New", 12))
age = tk.Entry(master=frame)
age.place(x=320, y=400)
res = tk.Label()
res.place(x=60, y=450)
res1 = tk.Label()
res1.place(x=50, y=470)
button_font = font.Font(family='Comfortaa', size=14, weight="normal")
button = tk.Button(master=frame, text="submit", bg='#45b592',
                   fg='#ffffff',
                   bd=0,
                   font=button_font,
                   height=2,
                   width=15, borderwidth=0)

button.place(x=400, y=500, width=100, height=40)
button.bind('<Button-1>', eventhandler)
window.mainloop()
