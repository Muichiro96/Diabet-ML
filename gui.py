import tkinter as tk

window = tk.Tk()
window.title("Diabetes analysis")
photo = tk.PhotoImage(file="diabetes-test.png")
window.wm_iconphoto(False, photo)

frame = tk.Frame(master=window, width=600, height=600)
frame.pack()


def eventhandler(e):
    print(entry1.get(), entry2.get(), entry21.get(), entry3.get(), entry5.get())


label1 = tk.Label(master=frame, text="Blood Pressure :")
label1.place(x=30, y=50)
entry1 = tk.Entry(master=frame)
entry1.place(x=130, y=50)
label2 = tk.Label(master=frame, text="Number Pregnacy :")
label2.place(x=30, y=100)
entry2 = tk.Entry(master=frame)
entry2.place(x=130, y=100)
label21 = tk.Label(master=frame, text="Insulin Level :")
label21.place(x=30, y=150)
entry21 = tk.Entry(master=frame)
entry21.place(x=130, y=150)
label3 = tk.Label(master=frame, text="Glucose Level :")
label3.place(x=30, y=200)
entry3 = tk.Entry(master=frame)
entry3.place(x=130, y=200)
label4 = tk.Label(master=frame, text="Skin Thickness :")
label4.place(x=30, y=250)
entry4 = tk.Entry(master=frame)
entry4.place(x=130, y=250)
label5 = tk.Label(master=frame, text="BodyMassIndex :")
label5.place(x=30, y=300)
entry5 = tk.Entry(master=frame)
entry5.place(x=130, y=300)
label6 = tk.Label(master=frame, text="Diabetes Pedigree Function :")
label6.place(x=30, y=350)
entry6 = tk.Entry(master=frame)
entry6.place(x=130, y=350)
label7 = tk.Label(master=frame, text="Age :")
label7.place(x=30, y=400)
entry7 = tk.Entry(master=frame)
entry7.place(x=130, y=400)
button = tk.Button(master=frame, text="Submit")
button.place(x=300, y=500, width=100, height=50)
button.bind('<Button-1>', eventhandler)
window.mainloop()
