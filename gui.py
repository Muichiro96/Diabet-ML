import tkinter as tk
from tkinter.ttk import Style

from model import Predictor

window = tk.Tk()
window.title("Diabetes analysis")
photo = tk.PhotoImage(file="diabetes-test.png")
window.wm_iconphoto(False, photo)

frame = tk.Frame(master=window, width=600, height=600)
frame.pack()
style = Style()
style.configure('TButton', font=
('calibri', 10, 'bold'),
                foreground='green')

predictor = Predictor()

def eventhandler(e):
    print(pregnancy.get(), glucoselevel.get(), bloodPressure.get(), skinThickness.get(), insulinLevel.get(), BMI.get(),
          DPF.get(), age.get())
    print(predictor.isDiabetic(getInputData()))
    # TODO check if true or false and show results to user!


def getInputData():
    return [pregnancy.get(), glucoselevel.get(), bloodPressure.get(), skinThickness.get(), insulinLevel.get(), BMI.get(),
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
button = tk.Button(master=frame, text="Submit")
button.place(x=300, y=500, width=100, height=50)
button.bind('<Button-1>', eventhandler)
window.mainloop()
