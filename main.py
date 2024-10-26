import tkinter as tk
from tkinter import filedialog
import PIL as pil
import PIL.ImageShow
from PIL import Image
from img_manipulation import LSB
from img_manipulation import DE

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

    # secret_img = LSB(image, secret_message).secret_image
    # decoded_message = LSB(secret_img).decoded_secret_message
    # print(decoded_message)
    # PIL.ImageShow.show(DE(image, secret_message).secret_image)
    secret = DE(image, secret_message)
    secret_img = secret.secret_image
    de = DE(secret_img, secret_message=None, lookup_string=secret.lookup_string)
    # print(secret.lookup_string)
    decoded_message, original_image= de.decoded_secret_message, de.original_image
    print(decoded_message)
    PIL.ImageShow.show(original_image)