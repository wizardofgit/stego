import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from img_manipulation import LSB, DE, PVD, calculate_mse, calculate_psnr

class SteganographyGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.text_file = None
        self.text_file_path = None
        self.image = None
        self.image_path = None

        self.title("Steganografia GUI")
        self.geometry("700x450")  # Adjusted window size
        self.configure(padx=20, pady=20)  # Padding for the main window

        # Frame for algorithm selection
        algorithm_frame = ttk.Frame(self)
        algorithm_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        tk.Label(algorithm_frame, text="Wybierz algorytm:").grid(row=0, column=0, sticky="w")

        self.algorithm_var = tk.StringVar()
        self.algorithm_menu = ttk.Combobox(
            algorithm_frame, textvariable=self.algorithm_var, values=["LSB", "DE", "PVD"], width=25
        )
        self.algorithm_menu.grid(row=0, column=1, sticky="ew")
        self.algorithm_menu.bind("<<ComboboxSelected>>", self.show_lookup_input)

        # Lookup string input for DE
        self.lookup_label = ttk.Label(algorithm_frame, text="Ciąg wyszukiwania:")
        self.lookup_entry = ttk.Entry(algorithm_frame, width=40)  # Larger lookup string input
        self.lookup_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.lookup_entry.grid(row=1, column=1, pady=(10, 0))
        self.lookup_label.grid_remove()
        self.lookup_entry.grid_remove()

        # Options frame for checkboxes
        options_frame = ttk.Frame(self)
        options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)

        # Save images checkbox
        self.save_images_var = tk.BooleanVar()
        self.save_images_checkbox = ttk.Checkbutton(
            options_frame, text="Zapisz obrazy", variable=self.save_images_var
        )
        self.save_images_checkbox.grid(row=0, column=0, sticky="w")

        # Display stats checkbox
        self.display_stats_var = tk.BooleanVar()
        self.display_stats_checkbox = ttk.Checkbutton(
            options_frame, text="Wyświetl statystyki", variable=self.display_stats_var
        )
        self.display_stats_checkbox.grid(row=0, column=1, sticky="w")

        # Display secret image checkbox
        self.display_secret_image_var = tk.BooleanVar()
        self.display_secret_image_checkbox = ttk.Checkbutton(
            options_frame, text="Wyświetl ukryty obraz", variable=self.display_secret_image_var
        )
        self.display_secret_image_checkbox.grid(row=0, column=2, sticky="w")

        # Button frame for file selection
        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)

        # Choose image button
        self.choose_image_button = ttk.Button(button_frame, text="Wybierz obraz", command=self.choose_image)
        self.choose_image_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Choose text file button
        self.choose_text_button = ttk.Button(button_frame, text="Wybierz plik .txt", command=self.choose_text_file)
        self.choose_text_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # General text input field
        text_input_frame = ttk.LabelFrame(self, text="Pole tekstowe")
        text_input_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=10)

        self.text_input = tk.Text(text_input_frame, height=6, wrap="word")
        self.text_input.pack(fill="both", expand=True, padx=10, pady=10)

        # Start embedding button
        self.start_button = ttk.Button(self, text="Rozpocznij osadzanie", command=self.start_embedding)
        self.start_button.grid(row=4, column=0, pady=10, sticky="ew")

        # Start decoding button
        self.start_decoding_button = ttk.Button(self, text="Rozpocznij dekodowanie", command=self.start_decoding)
        self.start_decoding_button.grid(row=4, column=1, pady=10, sticky="ew")

        # Configure grid weight for scalability
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(3, weight=1)

    def show_lookup_input(self, event=None):
        """Show or hide lookup input field based on algorithm selection."""
        if self.algorithm_var.get() == "DE":
            self.lookup_label.grid()
            self.lookup_entry.grid()
        else:
            self.lookup_label.grid_remove()
            self.lookup_entry.grid_remove()

    def choose_image(self):
        """Open file dialog to choose an image."""
        file_path = filedialog.askopenfilename(filetypes=[("Pliki graficzne", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.image_path = file_path
            messagebox.showinfo("Wybrano obraz", f"Wybrany obraz: {file_path}")

    def choose_text_file(self):
        """Open file dialog to choose a text file."""
        file_path = filedialog.askopenfilename(filetypes=[("Pliki tekstowe", "*.txt")])
        if file_path:
            self.text_file = open(file_path, "r", encoding="utf-8").read()
            self.text_file_path = file_path
            messagebox.showinfo("Wybrano plik tekstowy", f"Wybrany plik tekstowy: {file_path}")

    def start_embedding(self):
        """Handle the start embedding process and extract values from GUI components."""
        selected_algorithm = self.algorithm_var.get()
        text_input_content = self.text_input.get("1.0", tk.END).strip() if not self.text_file else self.text_file
        save_images = self.save_images_var.get()
        display_stats = self.display_stats_var.get()
        display_secret_image = self.display_secret_image_var.get()

        if selected_algorithm == "LSB":
            lsb = LSB(self.image, text_input_content)
            stego_image = lsb.secret_image

            if save_images:
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Pliki PNG", "*.png"), ("Wszystkie pliki", "*.*")])
                stego_image.save(file_path)
            if display_stats:
                messagebox.showinfo("Statystyki LSB", f"Czas: {lsb.time_diff:.2f} sekundy\n"
                                                     f"MSE: {calculate_mse(self.image, stego_image):.2f}\n"
                                                     f"PSNR: {calculate_psnr(self.image, stego_image):.2f}")
            if display_secret_image:
                stego_image.show()
        elif selected_algorithm == "DE":
            de = DE(self.image, text_input_content)
            stego_image = de.secret_image

            if save_images:
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Pliki PNG", "*.png"), ("Wszystkie pliki", "*.*")])
                stego_image.save(file_path)
            if display_stats:
                messagebox.showinfo("Statystyki DE", f"Czas: {de.time_diff:.2f} sekundy\n"
                                                     f"MSE: {calculate_mse(self.image, stego_image):.2f}\n"
                                                     f"PSNR: {calculate_psnr(self.image, stego_image):.2f}")
            if display_secret_image:
                stego_image.show()

            # Save the lookup string to a file
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")])
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(de.lookup_string)
        elif selected_algorithm == "PVD":
            pvd = PVD(self.image, text_input_content)
            stego_image = pvd.secret_image

            if save_images:
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Pliki PNG", "*.png"), ("Wszystkie pliki", "*.*")])
                stego_image.save(file_path)
            if display_stats:
                messagebox.showinfo("Statystyki PVD", f"Czas: {pvd.time_diff:.2f} sekundy\n"
                                                     f"MSE: {calculate_mse(self.image, stego_image):.2f}\n"
                                                     f"PSNR: {calculate_psnr(self.image, stego_image):.2f}")
            if display_secret_image:
                stego_image.show()

    def start_decoding(self):
        """Handle the start decoding process."""
        selected_algorithm = self.algorithm_var.get()
        lookup_string = self.lookup_entry.get() if selected_algorithm == "DE" else None
        save_images = self.save_images_var.get()

        if selected_algorithm == "LSB":
            decoded_text = LSB(self.image).decoded_secret_message

            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")])
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(decoded_text)
        elif selected_algorithm == "DE":
            de = DE(self.image, secret_message=None, lookup_string=lookup_string)
            decoded_text = de.decoded_secret_message
            original_image = de.original_image

            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")])
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(decoded_text)

            if save_images:
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Pliki PNG", "*.png"), ("Wszystkie pliki", "*.*")])
                original_image.save(file_path)
        elif selected_algorithm == "PVD":
            decoded_text = PVD(self.image).decoded_secret_message

            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")])
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(decoded_text)

if __name__ == "__main__":
    app = SteganographyGUI()
    app.mainloop()