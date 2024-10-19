import tkinter as tk
from tkinter import filedialog
import PIL as pil
from PIL import Image
from img_manipulation import LSB

if __name__ == '__main__':
    tk.Tk().withdraw()

    # select an image
    try:
        try:
            image_path = filedialog.askopenfilename(initialdir='./img/', title='Select an image')
        except:
            image_path = filedialog.askopenfilename(initialdir='./', title='Select an image')
        image = Image.open(image_path)
    except:
        raise Exception('Error: Image not selected or invalid image path')

    # select secret message file
    try:
        secret_message_path = filedialog.askopenfilename(initialdir='./', title='Select a secret message file', filetypes=[('text files', '*.txt')])
        with open(secret_message_path, 'rb') as file:
            secret_message = file.read().decode('utf-8') # read the secret message from the file as a string
    except:
        raise Exception('Error: Secret message not selected or invalid secret message path')

    # print(''.join(f'{byte:08b}' for byte in secret_message.encode('utf-8')))

    secret_img = LSB(image, secret_message).secret_image
    secret_img.show()