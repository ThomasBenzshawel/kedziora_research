import tkinter as tk
import sys
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from generator import generate

class MVPInterface:
    def __init__(self, master):
        self.master = master
        self.master.title("MVP")

        self.selected_images = []
        self.logs = []

        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.create_image_picker()
        self.create_log_display()
        self.create_generate_button()

        sys.stdout = LogRedirector(self.log_text)
        print("Initialized")

    def create_image_picker(self):
        self.btn_pick_image = ttk.Button(self.left_frame, text="Select Image", command=self.pick_image)
        self.btn_pick_image.pack(fill=tk.X, expand=True)

        self.image_group_box = ttk.LabelFrame(self.left_frame, text="Selected Images")
        self.image_group_box.pack(fill=tk.BOTH, expand=True)

        self.image_display_frame = tk.Frame(self.image_group_box)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True)

    def create_log_display(self):
        self.log_label = tk.Label(self.right_frame, text="Logs:")
        self.log_label.pack()

        self.log_text = tk.Text(self.right_frame, height=10, width=40)
        self.log_text.pack(fill=tk.Y, expand=True)

    def create_generate_button(self):
        self.btn_generate = ttk.Button(self.left_frame, text="Generate", command=self.generate, state=tk.DISABLED)
        self.btn_generate.pack(fill=tk.X, expand=True)

    def pick_image(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        for file_path in file_paths:
            if file_path:
                self.selected_images.append(file_path)
                self.display_images()
                print("Image: {}".format(file_path))
                self.btn_generate.config(state=tk.NORMAL)

    def display_images(self):
        for widget in self.image_display_frame.winfo_children():
            widget.destroy()

        for image_path in self.selected_images:
            image = Image.open(image_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self.image_display_frame, image=photo)
            label.image = photo # garbage collects otherwise
            label.pack()

        self.btn_generate.pack_forget()  # reset pos
        self.btn_generate.pack(fill=tk.X, expand=True) 

    def generate(self):
        print("Started Generating...")
        mesh = generate(self.selected_images)
        print("Done!")
        print("Creating Preview")
        mesh.show(viewer="gl")

    def log(self, message):
        self.logs.append(message)
        self.log_text.insert(tk.END, message + '\n')

class LogRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)

    def flush(self):
        pass

def main():
    root = tk.Tk()
    app = MVPInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()
