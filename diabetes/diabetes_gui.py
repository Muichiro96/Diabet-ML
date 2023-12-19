import tkinter as tk
from tkinter import font, messagebox
from diabetes.model import Predictor

predictor = Predictor()
window = tk.Tk()
window.title("Diabetes analysis")
window.resizable(width=False, height=False)
photo = tk.PhotoImage(file="diabetes-test.png")
window.wm_iconphoto(False, photo)
background_image = tk.PhotoImage(file="../bg4.png")
frame = tk.Frame(master=window, width=600, height=600)
frame.pack()
background_label = tk.Label(frame, image=background_image)
background_label.place(relwidth=1, relheight=1)


def validate_input():
    try:
        # Validate each input field as float
        int(pregnancy.get())
        float(glucoselevel.get())
        float(bloodPressure.get())
        float(skinThickness.get())
        float(insulinLevel.get())
        float(BMI.get())
        float(DPF.get())
        int(age.get())
        return True
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all input fields.")
        return False


def eventhandler(e):
    if not validate_input():
        return
    print(pregnancy.get(), glucoselevel.get(), bloodPressure.get(), skinThickness.get(), insulinLevel.get(), BMI.get(),
          DPF.get(), age.get())
    result, proba_false, proba_true = predictor.isDiabetic(getInputData())
    probas = f"Diabetic : {proba_true} \n Non Diabetic: {proba_false}"
    if result:
        res.config(text="Results :", font=("Courier New", 12),bg="#222",fg="white")
        res.place(x=60, y=450)
        res1.config(text="You are Diabetic \n" + probas, bg="red",
                    font="Helvetica 16 bold italic")
        res1.place(x=50, y=470)
    else:
        res.config(text="Results :", font=("Courier New", 12), bg="#222", fg="white")
        res.place(x=60, y=450)
        res1.config(text="You are not Diabetic \n" + probas, fg="light green", bg="dark green",
                    font="Helvetica 16 bold italic")
        res1.place(x=50, y=470)


def getInputData():
    return [pregnancy.get(), glucoselevel.get(), bloodPressure.get(), skinThickness.get(), insulinLevel.get(),
            BMI.get(),
            DPF.get(), age.get()]


label1 = tk.Label(master=frame, text="Blood Pressure :")
label1.place(x=30, y=50)
label1.config(font=("Courier New", 12), bg="#222", fg="white")
bloodPressure = tk.Entry(master=frame)
bloodPressure.place(x=320, y=50)
label2 = tk.Label(master=frame, text="Number of pregnacies :")
label2.place(x=30, y=100)
label2.config(font=("Courier New", 12), bg="#222", fg="white")
pregnancy = tk.Entry(master=frame)
pregnancy.place(x=320, y=100)
label21 = tk.Label(master=frame, text="Insulin Level :")
label21.place(x=30, y=150)
label21.config(font=("Courier New", 12), bg="#222", fg="white")
insulinLevel = tk.Entry(master=frame)
insulinLevel.place(x=320, y=150)
label3 = tk.Label(master=frame, text="Glucose Level :")
label3.place(x=30, y=200)
label3.config(font=("Courier New", 12), bg="#222", fg="white")
glucoselevel = tk.Entry(master=frame)
glucoselevel.place(x=320, y=200)
label4 = tk.Label(master=frame, text="Skin Thickness :")
label4.place(x=30, y=250)
label4.config(font=("Courier New", 12), bg="#222", fg="white")
skinThickness = tk.Entry(master=frame)
skinThickness.place(x=320, y=250)
label5 = tk.Label(master=frame, text="BodyMassIndex :")
label5.place(x=30, y=300)
label5.config(font=("Courier New", 12), bg="#222", fg="white")
BMI = tk.Entry(master=frame)
BMI.place(x=320, y=300)
label6 = tk.Label(master=frame, text="Diabetes Pedigree Function :")
label6.place(x=30, y=350)
label6.config(font=("Courier New", 12), bg="#222", fg="white")
DPF = tk.Entry(master=frame)
DPF.place(x=320, y=350)
label7 = tk.Label(master=frame, text="Age :")
label7.place(x=30, y=400)
label7.config(font=("Courier New", 12), bg="#222", fg="white")
age = tk.Entry(master=frame)
age.place(x=320, y=400)
res = tk.Label()

res1 = tk.Label()

button_font = font.Font(family='Comfortaa', size=14, weight="normal")
button = tk.Button(master=frame, text="submit", bg='#45b592',
                   fg='#ffffff',
                   bd=0,
                   font=button_font,
                   height=2,
                   width=15, borderwidth=0)

button.place(x=450, y=500, width=100, height=40)
button.bind('<Button-1>', eventhandler)
window.mainloop()
