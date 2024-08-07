import concurrent.futures
from PIL import Image, ImageTk
import os
import numpy as np
import tkinter as tk
from tkinter import Scale, Button, Scrollbar, Listbox, Entry
import cv2
import hashlib
import concurrent
import threading
from io import BytesIO
import requests

class GUI:
    def __init__(self, root):
        self.root = root
        self.path = "./"
        self.images = [image for image in os.listdir(self.path) if image.endswith(('webp', 'jpg', 'jpeg', 'png', 'bmp'))]
        self.images.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isnumeric() else x)
        self.current_image_index = 0
        self.init_ui()
        self.transformed = 0
        self.original = 0
        self.kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.original_img = None
        self.edge = None
        self.transformed_img = None
        self.region_mask = None
        self.shape = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.run = False
        self.condition = threading.Condition()
        self.last_task = None
        self.blur = {}
        self.interpolated_img = None
    
    def __del__(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def init_ui(self):
        self.root.title("Image Processor")
        
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(main_frame, bg='gray')
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)
        
        self.loading_label = tk.Label(right_frame, text="", font=("Helvetica", 12))
        self.loading_label.grid(row=0, column=0, pady=10)
        
        slinder = tk.Frame(right_frame)
        slinder.grid(row=1, column=0, pady=10)
        
        self.min_slider = Scale(slinder, from_=0, to=255, orient=tk.HORIZONTAL, label='Min', command=lambda val: self.update_image(int(val), self.max_slider.get(), self.strength_slider.get()))
        self.min_slider.pack(side=tk.LEFT)
        
        self.max_slider = Scale(slinder, from_=0, to=255, orient=tk.HORIZONTAL, label='Max', command=lambda val: self.update_image(self.min_slider.get(), int(val), self.strength_slider.get()))
        self.max_slider.set(255)
        self.max_slider.pack(side=tk.LEFT)

        sub_frame = tk.Frame(right_frame)
        sub_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.strength_slider = Scale(sub_frame, from_=1, to=11, orient=tk.HORIZONTAL, label='Strength', command=lambda val: self.update_image(self.min_slider.get(), self.max_slider.get(), int(val)))
        self.strength_slider.set(6)
        self.strength_slider.grid(row=2, column=0, padx=(50, 0))
        
        self.fastmode = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(sub_frame, text="Fast mode", variable=self.fastmode, command=lambda: self.blur.clear())
        self.checkbox.grid(row=2, column=1, pady=(30, 0))
        self.checkbox.select()

        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Left>', self.before_image)
        
        rb_frame = tk.Frame(right_frame)
        rb_frame.grid(row=3, column=0, pady=10)
        
        self.next_button = Button(rb_frame, text='Next', padx=10, command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)
        
        self.save_button = Button(rb_frame, text='Save', padx=10, command=self.save_image)
        self.save_button.pack(side=tk.RIGHT)
        
        entry_frame = tk.Frame(right_frame)
        entry_frame.grid(row=4, column=0, pady=10, sticky="ew")
        
        entry_label = tk.Label(entry_frame, text="경로")
        entry_label.pack(side=tk.LEFT)
        
        self.entry = Entry(entry_frame, width=40)
        self.entry.pack(side=tk.LEFT, padx=10)
        
        listbox_frame = tk.Frame(right_frame)
        listbox_frame.grid(row=5, column=0, pady=10, sticky="nsew")
        
        self.scrollbar = Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.listbox = Listbox(listbox_frame, width=40, height=15, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        buttons_frame = tk.Frame(right_frame)
        buttons_frame.grid(row=6, column=0, pady=10, sticky="ew")
        
        self.add_button = Button(buttons_frame, text="새로고침", command=self.add_path)
        self.add_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = Button(buttons_frame, text="선택", command=self.process_selected_path)
        self.process_button.pack(side=tk.RIGHT, padx=5)

    def next_image(self, event=None):
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.load_image(os.path.join(self.path, self.images[self.current_image_index]))

    def before_image(self, event=None):
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.load_image(os.path.join(self.path, self.images[self.current_image_index]))

    def add_path(self):
        self.path = self.entry.get()
        self.images = []
        if not self.path:
            self.path = "./"

        if os.path.isfile(self.path) and self.path.endswith(('webp', 'jpg', 'jpeg', 'png', 'bmp')):
            self.images = [os.path.basename(self.path)]
            self.path = os.path.dirname(self.path)
        elif os.path.isdir(self.path):
            self.images = [image for image in os.listdir(self.path) if image.endswith(('webp', 'jpg', 'jpeg', 'png', 'bmp'))]
            self.images.sort(key=lambda x: os.path.splitext(x)[0].isnumeric() and (int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isnumeric() else x))
        elif self.path.startswith("http"):
            try:
                response = requests.head(self.path, allow_redirects=True)
                content_type = response.headers.get("Content-Type", '')
                if content_type.startswith('image/'):
                    self.images = [self.path]
                    self.path = ""
            except:
                pass

        self.listbox.delete(0, tk.END)
        for image in self.images:
            self.listbox.insert(tk.END, image)

    def process_selected_path(self):
        selected_index = self.listbox.curselection()
        if selected_index:
            self.current_image_index = selected_index[0]
            self.load_image(os.path.join(self.path, self.images[self.current_image_index]))

    def adjust_rgb(self, x: np.ndarray):
        r = np.where(x[:, :, 0] < 0, -x[:, :, 0], 0)
        g = np.where(x[:, :, 1] < 0, -x[:, :, 1], 0)
        b = np.where(x[:, :, 2] < 0, -x[:, :, 2], 0)

        x += (r + g + b).reshape((*x.shape[:2], 1))
        return x
    
    def apply_convolution(self):
        img_height, img_width, img_channel = self.original_img.shape
        kernel_height, kernel_width = self.kernel.shape

        padding = (kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)
        padded_img = np.pad(self.original_img, (*padding, (0, 0)), 'edge')
        
        window_shape = (img_height, img_width, kernel_height, kernel_height, img_channel)
        strides = padded_img.strides[:2] + padded_img.strides[:2] + padded_img.strides[2:]
        sliding_windows = np.lib.stride_tricks.as_strided(padded_img, shape=window_shape, strides=strides)
        
        transformed_img = np.tensordot(sliding_windows, self.kernel, axes=((2, 3), (0, 1)))
        transformed_img = self.adjust_rgb(transformed_img)
        self.edge = np.clip(transformed_img, 0, 255).astype(np.uint8)

    def image_process(self, min_val, max_val, strength):
        if self.edge is None:
            return
        
        self.run = True
        if self.original_img is not None:
            self.loading_label.config(text="작업 중...")
        
        max_val = max(min(max_val, 255), 1)
        min_val = min(max_val - 1, max(0, min_val))

        transformed_img = np.clip(self.edge, min_val, max_val)
        self.region_mask = (self.edge - transformed_img).astype(bool)
        transformed_img = transformed_img - min_val
        self.transformed_img = 255 - (transformed_img.astype(np.float64) * 255 / (max_val - min_val)).astype(np.uint8)
        self.photo_transformed = ImageTk.PhotoImage(Image.fromarray(self.transformed_img).resize(self.shape))
        self.canvas.itemconfig(self.transformed, image=self.photo_transformed)

        if self.blur.get(strength) is None:
            self.blur[strength] = cv2.bilateralFilter(self.original_img, -1 if not self.fastmode.get() else 17, strength, strength)

        self.interpolated_img = self.original_img.copy()
        self.interpolated_img[self.region_mask] = self.blur[strength][self.region_mask]

        self.photo_original = ImageTk.PhotoImage(Image.fromarray(self.interpolated_img).resize(self.shape))
        self.canvas.itemconfig(self.original, image=self.photo_original)

        self.loading_label.config(text="")

    def run_task(self, func, *args):
        try:
            func(*args)
        finally:
            self.run = False
            with self.condition:
                self.condition.notify()

    def update_image(self, min_val, max_val, strength):
        with self.condition:
            args = min_val, max_val, 20 + 5 * strength
            if self.run:
                self.last_task = self.image_process, args
            else:
                future = self.executor.submit(self.run_task, self.image_process, *args)
                future.add_done_callback(self.task_done)

    def task_done(self, future):
        with self.condition:
            if self.last_task:
                func, args = self.last_task
                self.last_task = None
                future = self.executor.submit(self.run_task, func, *args)
                future.add_done_callback(self.task_done)

    def load_image(self, image_path:str):
        if image_path:
            self.loading_label.config(text="작업 중...")
        
        if os.path.isfile(image_path):
            self.original_img = np.array(Image.open(image_path).convert("RGB"))
        else:
            response = requests.get(image_path)
            self.original_img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        ratio = self.original_img.shape[1], self.original_img.shape[0]
        self.blur.clear()

        self.shape = (768 * ratio[0] // max(ratio), 768 * ratio[1] // max(ratio))
        self.apply_convolution()
        
        transformed_pil_image = Image.fromarray(255 - self.edge)
        photo_transformed = transformed_pil_image.resize(self.shape)

        self.photo_transformed = ImageTk.PhotoImage(photo_transformed)
        self.canvas.config(width=2*self.shape[0], height=self.shape[1])
        self.root.update_idletasks()

        ypos = self.canvas.winfo_height() / 2 - self.shape[1] / 2
        self.transformed = self.canvas.create_image(self.shape[0], ypos, anchor=tk.NW, image=self.photo_transformed)

        original_pil_image = Image.fromarray(self.original_img)
        photo_original = original_pil_image.resize(self.shape)

        self.photo_original = ImageTk.PhotoImage(photo_original)
        self.original = self.canvas.create_image(0, ypos, anchor=tk.NW, image=self.photo_original)
        self.update_image(self.min_slider.get(), self.max_slider.get(), self.strength_slider.get())
        self.loading_label.config(text="")

    def save_image(self):
        img = Image.fromarray(self.interpolated_img)
        img.save(hashlib.sha256(img.tobytes()).hexdigest() + '.webp', format='WEBP')
        img = Image.fromarray(self.transformed_img).convert('L')
        img.save(hashlib.sha256(img.tobytes()).hexdigest() + '.webp', format='WEBP')

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()