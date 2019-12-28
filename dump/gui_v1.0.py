import os
import webbrowser
import numpy as np
import csv
import traceback
import arabic_reshaper
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from run_model import create_and_run_model

def make_menu(w):
    global the_menu
    the_menu = Menu(w, tearoff=0)
    the_menu.add_command(label="Cut")
    the_menu.add_command(label="Copy")
    the_menu.add_command(label="Paste")

def show_menu(e):
    w = e.widget
    the_menu.entryconfigure("Cut", command=lambda: w.event_generate("<<Cut>>"))
    the_menu.entryconfigure("Copy", command=lambda: w.event_generate("<<Copy>>"))
    the_menu.entryconfigure("Paste", command=lambda: w.event_generate("<<Paste>>"))
    the_menu.tk.call("tk_popup", the_menu, e.x_root, e.y_root)

def main_window(): 
    window = Tk()
    make_menu(window)
    window.title("Urdu Handwriting Recognition System")
    window.geometry('1000x1000')
    title = Label(window, text="Urdu Handwriting Recognition System", font=("Arial Bold", 30))
    title.grid(column=1, row=0, columnspan=10)
    window.grid_rowconfigure(0, minsize=100)
    window.grid_rowconfigure(1, minsize=70)
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(11, weight=1)

    window.grid_columnconfigure(11, weight=1)
    
    col_path = 3
    row_path = 2

    display_path = Label(window, text="Enter Image Path: ")
    display_path.grid(column=col_path, row=row_path)
    
    window.grid_rowconfigure(row_path+1, minsize=50)
    window.grid_rowconfigure(row_path+2, minsize=100)
    
    display_image = Label(window, image='')
    display_image.grid(column=col_path-2, row=row_path+2, columnspan=10)
    
    display_raw_output = Label(window, text='', font=("Arial Bold", 15))
    display_raw_output.grid(column= col_path-2, row=row_path+3, columnspan=10)
    window.grid_rowconfigure(row_path+3, minsize=60)

    #display_output = Label(window, text='', font=("Arial Bold", 15))
    display_output = Entry(window, width=40, justify='right')
    display_output.bind_class("Entry", "<Button-3><ButtonRelease-3>", show_menu)
    display_output.grid(column= col_path-2, row=row_path+4, columnspan=10)

    get_image_path = Entry(window,width=40)
    get_image_path.bind_class("Entry", "<Button-3><ButtonRelease-3>", show_menu)
    get_image_path.grid(column=col_path+1, row=row_path)
    get_image_path.focus()
    
    def select():
        image_path = askopenfilename()
        get_image_path.delete(0, END)
        get_image_path.insert(0, image_path)
        
        img = ImageTk.PhotoImage(Image.open(image_path))
        display_image.configure(image = img)
        display_image.image = img
        display_raw_output.configure(text = '')
        #display_output.configure(text = '')
        display_output.delete(0, END)

    def clicked():
        image_path = get_image_path.get()

        if image_path is '':
            messagebox.showinfo("Error", "Select an image")
        elif os.path.isfile(image_path) == False:
            messagebox.showinfo("Error", "File does not exist")
        else:
            img = ImageTk.PhotoImage(Image.open(image_path))
            display_image.configure(image = img)
            display_image.image = img
 
            output = create_and_run_model('CONV_BLSTM_CTC', None, image_path)
            raw_output, join_char = get_urdu_output(output)
            with open("output.txt", "w") as text_file:
                text_file.write("%s" % join_char)

            webbrowser.open("output.txt")
            #with open("output.txt", "r") as text_file:
            #    join_char = text_file.read().replace('\n', '')

            display_raw_output.configure(text = raw_output)
            #display_output.configure(text = join_char)
            display_output.delete(0, END)
            display_output.insert(0, join_char)
       
            #with open("output.csv", mode='w') as f:
            #    f_w = csv.writer(f, delimiter=',')
            #    f_w.writerow(join_char)

    browse = Button(window, text="Browse", command=select)
    browse.grid(column=col_path+2, row=row_path)

    recognize = Button(window, text="Recognize", command=clicked)
    recognize.grid(column=col_path+3, row=row_path)

    window.mainloop()

def get_urdu_output(output):
    lt_file = 'data/segmented_cc/labels/lt_char.csv'
    lt = {}
    with open(lt_file, 'r', encoding='utf8') as file:
        text = csv.reader(file)
        for row in text:
           lt[int(row[1])] = row[0]
    
    urdu_output = [lt[output[i]] for i in range(len(output)-1, -1, -1)]
    
    join_char = ''
    for i in range(len(urdu_output)-1, -1, -1):
    #for i in range(0, len(urdu_output)):
        join_char += urdu_output[i][0]
        if urdu_output[i][2:] == 'final' or urdu_output[i][2:] == 'isolated':
            join_char += ' '
    
   #join_char = arabic_reshaper.reshape(join_char)

    return urdu_output, join_char

if __name__ == "__main__":
    main_window()
