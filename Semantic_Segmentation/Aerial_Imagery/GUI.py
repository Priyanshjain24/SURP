# Python program to create
# a file explorer in Tkinter
  
# import all components
# from the tkinter library
from tkinter import *
  
# import filedialog module
from tkinter import filedialog
from run_model import *
import matplotlib.pyplot as plt

window = Tk()
image1 = StringVar(window, name ="im1")
folder = StringVar(window, name ="fol")

def browseFiles():
    filename = filedialog.askopenfilenames(initialdir = "/home",
                                          title = "Select two Images",
                                          filetypes = (
                                                       ("all files",
                                                        "*.*"), ("Text files","*.txt*")))
    print(filename)
    if len(filename) == 1 and type(filename) == tuple:
        window.setvar(name ="im1", value =filename[0])
        label_file1_explorer.configure(text="Input image 1:" + str(image1.get()))
    
def browseFolder():
    oupt = filedialog.askdirectory(initialdir = "/home",
                                          title = "Select a Folder",
                                          )
    window.setvar(name="fol", value=oupt)
    label_folder_explorer.configure(text="Output folder:" + folder.get())

def runmodel():
    window.withdraw()
    
    output_img = predict(image1.get())
    folder_nm = folder.get()
    plt.imsave(folder_nm + "/output.png", output_img)
    
    window.quit()
    

# Create the root window
  
# Set window title
window.title('File Explorer')
  
# Set window size
window.geometry("500x500")
  
#Set window background color
window.config(background = "white")
  
# Create a File Explorer label
label_file_explorer = Label(window,
                            text = "File Explorer using Tkinter",
                            width = 100, height = 4,
                            fg = "blue")

label_file1_explorer = Label(window,
                            text = "Input image :",
                            width = 100, height = 4,
                            fg = "blue")

label_folder_explorer = Label(window,
                            text = "Output folder:",
                            width = 100, height = 4,
                            fg = "blue")
  
      
button_explore = Button(window,
                        text = "Input Image",
                        command = browseFiles)

button_folder_explore = Button(window, text="Output directory", command=browseFolder)
  
button_run = Button(window,
                     text = "Run",
                     command = runmodel)

button_exit = Button(window,
                     text = "Exit",
                     command = exit)
  
# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column = 1, row = 1)
  
button_explore.grid(column = 1, row = 2)
button_folder_explore.grid(column = 1, row = 3)  
label_file1_explorer.grid(column = 1, row = 4)
label_folder_explorer.grid(column = 1, row = 6)
button_run.grid(column=1, row=7)
button_exit.grid(column = 1,row = 8)
  
# Let the window wait for any events
window.mainloop()
