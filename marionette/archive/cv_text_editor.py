import tkinter as tk
from tkinter import colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk

def update_image():
    global img_display
    if not cap.isOpened():
        return
    ret, frame = cap.read()
    if not ret:
        return

    text = text_var.get()
    x_pos = int(x_pos_var.get())
    y_pos = int(y_pos_var.get())
    font = font_dict[font_var.get()]
    font_scale = float(font_scale_var.get())
    color = tuple(int(col) for col in color_var.get().split(','))
    thickness = int(thickness_var.get())

    cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
    img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    canvas.itemconfig(image_on_canvas, image=img_display)
    root.after(10, update_image)  # Refresh the image on the canvas every 10 milliseconds

def choose_color():
    color_code = colorchooser.askcolor(title="Choose text color")
    if color_code:
        color_var.set(str(int(color_code[0][0])) + ',' + str(int(color_code[0][1])) + ',' + str(int(color_code[0][2])))
        update_image()

# Create the main window
root = tk.Tk()
root.title("OpenCV Text Configuration with Webcam")

# Variables
text_var = tk.StringVar(value='Initial Text')
x_pos_var = tk.StringVar(value='10')
y_pos_var = tk.StringVar(value='60')
font_scale_var = tk.StringVar(value='1')
color_var = tk.StringVar(value='0,0,0')
thickness_var = tk.StringVar(value='2')
font_var = tk.StringVar(value='HERSHEY_SIMPLEX')

# Dictionary of fonts
font_dict = {
    'HERSHEY_SIMPLEX': cv2.FONT_HERSHEY_SIMPLEX,
    'HERSHEY_PLAIN': cv2.FONT_HERSHEY_PLAIN,
    'HERSHEY_DUPLEX': cv2.FONT_HERSHEY_DUPLEX,
    'HERSHEY_COMPLEX': cv2.FONT_HERSHEY_COMPLEX,
    'HERSHEY_TRIPLEX': cv2.FONT_HERSHEY_TRIPLEX,
    'HERSHEY_COMPLEX_SMALL': cv2.FONT_HERSHEY_COMPLEX_SMALL,
    'HERSHEY_SCRIPT_SIMPLEX': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    'HERSHEY_SCRIPT_COMPLEX': cv2.FONT_HERSHEY_SCRIPT_COMPLEX
}

# Set up the GUI
tk.Label(root, text="Text:").grid(row=0, column=0)
tk.Entry(root, textvariable=text_var, width=10).grid(row=0, column=1)

tk.Label(root, text="X Position:").grid(row=1, column=0)
tk.Entry(root, textvariable=x_pos_var, width=10).grid(row=1, column=1)

tk.Label(root, text="Y Position:").grid(row=2, column=0)
tk.Entry(root, textvariable=y_pos_var, width=10).grid(row=2, column=1)

tk.Label(root, text="Font Scale:").grid(row=3, column=0)
tk.Entry(root, textvariable=font_scale_var, width=10).grid(row=3, column=1)

tk.Label(root, text="Font Type:").grid(row=4, column=0)
tk.OptionMenu(root, font_var, *font_dict.keys()).grid(row=4, column=1)

tk.Label(root, text="Color:").grid(row=5, column=0)
tk.Entry(root, textvariable=color_var, width=10).grid(row=5, column=1)
tk.Button(root, text="Choose Color", command=choose_color).grid(row=5, column=2)

tk.Label(root, text="Thickness:").grid(row=6, column=0)
tk.Entry(root, textvariable=thickness_var, width=10).grid(row=6, column=1)

# Webcam capture setup
cap = cv2.VideoCapture(0)  # Open default camera

# Setup for image display
canvas = tk.Canvas(root, width=640, height=480)  # Adjust size as needed based on your webcam's resolution
canvas.grid(row=7, column=0, columnspan=3)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=None)

# Start the GUI loop
root.after(10, update_image)  # Start the periodic update of webcam images
root.mainloop()

# Release the webcam on closing
cap.release()
cv2.destroyAllWindows()
