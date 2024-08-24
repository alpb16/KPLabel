import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# List to hold keypoints, bounding box, and actions
keypoints = []
bounding_box = []
actions = []  # List to track actions
image_paths = []
current_image_index = 0
image = None
original_image = None
scaling_factor = 1.0
mode = "point"  # Default mode
save_directory = "."  # Default save directory

# Create the Tkinter root window first
root = tk.Tk()
root.title("Image Keypoint Labeling")

# Variable for Auto Box checkbox
auto_box_enabled = tk.BooleanVar()


def on_canvas_click(event):
    global bounding_box, keypoints, mode, actions, image
    x, y = int(event.x / scaling_factor), int(event.y / scaling_factor)
    if mode == "box":
        if len(bounding_box) == 0:
            # Start the bounding box selection
            bounding_box.append((x, y))
            canvas.bind("<Motion>", on_mouse_move)  # Bind mouse motion for preview
        elif len(bounding_box) == 1:
            # Complete the bounding box selection
            bounding_box.append((x, y))
            # Clear previous preview
            canvas.delete("preview_box")
            draw_bounding_box()
            actions.append(('box', bounding_box.copy()))
            canvas.unbind("<Motion>")
    elif mode == "point":
        keypoints.append((x, y))
        actions.append(('point', (x, y)))
        draw_keypoint(x, y)
        if auto_box_enabled.get():  # Update bounding box if auto_box is enabled
            update_image_with_points()
            auto_box()

def create_thumbnail(image_path, size=(100, 100)):
    """Create a thumbnail of the image."""
    img = Image.open(image_path)
    img.thumbnail(size)
    return ImageTk.PhotoImage(img)


def update_gallery():
    """Update the gallery with thumbnails of all images."""
    global gallery_frame_inner, image_paths, gallery_buttons, canvas

    # Clear the existing gallery
    for widget in gallery_frame_inner.winfo_children():
        widget.destroy()

    gallery_buttons = []
    for path in image_paths:
        thumbnail = create_thumbnail(path)

        # Create a frame for the thumbnail and its label
        frame = tk.Frame(gallery_frame_inner, bg='white')
        frame.pack(pady=5, anchor='w')

        # Create a button for the thumbnail
        button = tk.Button(frame, image=thumbnail, command=lambda p=path: load_image(p))
        button.image = thumbnail  # Keep a reference to avoid garbage collection
        button.pack()

        # Create a label for the image name
        label = tk.Label(frame, text=os.path.basename(path))
        label.pack()

        gallery_buttons.append(button)

        # Update the scroll region
    canvas.config(scrollregion=canvas.bbox("all"))


def on_mouse_move(event):
    global bounding_box, image

    if len(bounding_box) == 1:
        # Calculate the current position to show the preview
        x, y = int(event.x / scaling_factor), int(event.y / scaling_factor)
        temp_box = [bounding_box[0], (x, y)]

        # Clear previous preview
        canvas.delete("preview_box")

        # Draw preview box
        x1, y1 = temp_box[0]
        x2, y2 = temp_box[1]
        canvas.create_rectangle(x1 * scaling_factor, y1 * scaling_factor,
                                x2 * scaling_factor, y2 * scaling_factor,
                                outline="red", width=2, tag="preview_box")

def calculate_bbox_details(box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return center_x, center_y, width, height


def draw_bounding_box():
    global image, bounding_box, canvas, image_on_canvas
    canvas.delete("bounding_box")  # Remove previous bounding box
    if bounding_box:
        x1, y1 = bounding_box[0]
        x2, y2 = bounding_box[1]
        cv2.rectangle(image, (int(x1 * scaling_factor), int(y1 * scaling_factor)),
                      (int(x2 * scaling_factor), int(y2 * scaling_factor)), (0, 0, 255), 2)
        show_image()


def draw_keypoint(x, y):
    global image, canvas, image_on_canvas
    cv2.circle(image, (int(x * scaling_factor), int(y * scaling_factor)), 3, (0, 255, 0), -1)
    show_image()


def resize_image_to_fit(image, max_width, max_height):
    global scaling_factor
    height, width, _ = image.shape
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(image, (new_width, new_height)), scaling_factor


def load_image(image_path):
    global image, original_image, keypoints, bounding_box, scaling_factor, actions
    keypoints = []
    bounding_box = []
    actions = []  # Reset actions list when a new image is loaded
    original_image = cv2.imread(image_path)
    image, scaling_factor = resize_image_to_fit(original_image, canvas.winfo_width(), canvas.winfo_height())
    load_annotations(image_path)
    show_image()


def show_image():
    global image, canvas, image_on_canvas
    bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(bgr_image)
    tk_image = ImageTk.PhotoImage(pil_image)

    # Check if image is already on canvas and update it
    if 'image_on_canvas' in globals():
        canvas.itemconfig(image_on_canvas, image=tk_image)
    else:
        image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.image = tk_image


def save_annotations():
    global bounding_box, keypoints, original_image, save_directory

    # Check if a bounding box exists; if not, create one
    if len(bounding_box) != 2 and keypoints:
        auto_box()

    # If still no bounding box, skip saving
    if len(bounding_box) != 2:
        return

    height, width, _ = original_image.shape
    object_label = 0  # Example object label
    txt_file_name = os.path.join(save_directory,
                                 f"{os.path.splitext(os.path.basename(image_paths[current_image_index]))[0]}.txt")

    with open(txt_file_name, "w") as file:
        if len(bounding_box) == 2:
            center_x, center_y, bbox_width, bbox_height = calculate_bbox_details(bounding_box)
            file.write(
                f"{object_label} {center_x / width} {center_y / height} {bbox_width / width} {bbox_height / height} ")

        for (x, y) in keypoints:
            normalized_x = x / width
            normalized_y = y / height
            file.write(f"{normalized_x} {normalized_y} ")


def load_annotations(image_path):
    global bounding_box, keypoints, original_image, actions
    height, width, _ = original_image.shape
    txt_file_name = os.path.join(save_directory, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    if os.path.exists(txt_file_name):
        with open(txt_file_name, "r") as file:
            data = file.read().split()
            if len(data) >= 5:
                center_x = float(data[1]) * width
                center_y = float(data[2]) * height
                bbox_width = float(data[3]) * width
                bbox_height = float(data[4]) * height
                x1 = center_x - bbox_width / 2
                y1 = center_y - bbox_height / 2
                x2 = center_x + bbox_width / 2
                y2 = center_y + bbox_height / 2
                bounding_box = [(int(x1), int(y1)), (int(x2), int(y2))]
                actions.append(('box', bounding_box.copy()))
                draw_bounding_box()

            keypoints.clear()
            for i in range(5, len(data), 2):
                x = float(data[i]) * width
                y = float(data[i + 1]) * height
                keypoints.append((int(x), int(y)))
                actions.append(('point', (int(x), int(y))))
                draw_keypoint(int(x), int(y))


def next_image():
    global current_image_index, image_paths, keypoints
    if keypoints:  # If keypoints exist, draw the bounding box before moving to the next image
        auto_box()
    save_annotations()  # Save the current annotations before moving to the next image
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        load_image(image_paths[current_image_index])



def previous_image():
    global current_image_index, image_paths
    if current_image_index > 0:
        current_image_index -= 1
        load_image(image_paths[current_image_index])


def reset_image():
    global keypoints, bounding_box, image_paths, actions
    keypoints = []
    bounding_box = []
    actions = []
    load_image(image_paths[current_image_index])

    # Delete the text file
    txt_file_name = os.path.join(save_directory,
                                 f"{os.path.splitext(os.path.basename(image_paths[current_image_index]))[0]}.txt")
    if os.path.exists(txt_file_name):
        os.remove(txt_file_name)

def select_directory():
    global image_paths, current_image_index, save_directory
    directory = filedialog.askdirectory()
    if directory:
        save_directory = directory  # Update the save directory
        image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png'))]
        if image_paths:
            current_image_index = 0
            load_image(image_paths[current_image_index])
            update_gallery()  # Update the gallery


def create_thumbnail(image_path, size=(100, 100)):
    """Create a thumbnail of the image."""
    img = Image.open(image_path)
    img.thumbnail(size)
    return ImageTk.PhotoImage(img)


def update_gallery():
    """Update the gallery with thumbnails of all images."""
    global gallery_frame, image_paths, gallery_buttons

    # Clear the existing gallery
    for widget in gallery_frame.winfo_children():
        widget.destroy()

    gallery_buttons = []
    for path in image_paths:
        # Create a thumbnail
        thumbnail = create_thumbnail(path)

        # Create a frame for the thumbnail and its label
        frame = tk.Frame(gallery_frame)
        frame.pack(pady=5)

        # Create a button for the thumbnail
        button = tk.Button(frame, image=thumbnail, command=lambda p=path: load_image(p))
        button.image = thumbnail  # Keep a reference to avoid garbage collection
        button.pack()

        # Create a label for the image name
        label = tk.Label(frame, text=os.path.basename(path))
        label.pack()

        gallery_buttons.append(button)


def set_mode(new_mode):
    global mode
    mode = new_mode


def auto_box():
    global bounding_box, keypoints, image
    if not keypoints:
        return
    x_coords = [p[0] for p in keypoints]
    y_coords = [p[1] for p in keypoints]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    bounding_box = [(x1, y1), (x2, y2)]
    actions.append(('box', bounding_box.copy()))
    draw_bounding_box()


def update_image_with_points():
    global keypoints, image, scaling_factor
    keypoints_temp = keypoints
    reset_image()
    keypoints = keypoints_temp
    # Redraw all points
    for x, y in keypoints:
        draw_keypoint(x, y)
    # Redraw bounding box if it exists
    if bounding_box:
        draw_bounding_box()
    show_image()


def resize_canvas(event):
    if image_paths:
        canvas_width = event.width
        canvas_height = event.height
        load_image(image_paths[current_image_index])

def bind_keys():
    root.bind('o', lambda event: select_directory())
    root.bind('<Left>', lambda event: previous_image())
    root.bind('a', lambda event: previous_image())
    root.bind('<Right>', lambda event: next_image())
    root.bind('d', lambda event: next_image())
    root.bind('r', lambda event: reset_image())
    root.bind('s', lambda event: save_annotations())
    root.bind('z', lambda event: set_mode("point"))
    root.bind('x', lambda event: set_mode("box"))
    root.bind('b', lambda event: toggle_auto_box())

def toggle_auto_box():
    auto_box_enabled.set(not auto_box_enabled.get())


















# Frame for buttons and gallery (buttons on top, gallery at the bottom)
button_and_gallery_frame = tk.Frame(root)
button_and_gallery_frame.pack(side=tk.RIGHT, fill=tk.Y)



# Create a frame for the buttons and canvas
frame = tk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Canvas for displaying the image
canvas = tk.Canvas(frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(button_and_gallery_frame)
button_frame.pack(side=tk.TOP, fill=tk.X)

select_dir_button = tk.Button(button_frame, text="Select Directory (o)", command=select_directory)
select_dir_button.pack(pady=5)

previous_image_button = tk.Button(button_frame, text="Previous Image (<- or a)", command=previous_image)
previous_image_button.pack(pady=5)

next_image_button = tk.Button(button_frame, text="Next Image (-> or d)", command=next_image)
next_image_button.pack(pady=5)





reset_button = tk.Button(button_frame, text="Reset Image (r)", command=reset_image)
reset_button.pack(pady=5)

save_annotations_button = tk.Button(button_frame, text="Save Annotations (s)", command=save_annotations)
save_annotations_button.pack(pady=5)

point_button = tk.Button(button_frame, text="Point Mode (z)", command=lambda: set_mode("point"))
point_button.pack(pady=5)

box_button = tk.Button(button_frame, text="Box Mode (x)", command=lambda: set_mode("box"))
box_button.pack(pady=5)

# Frame for gallery
gallery_frame = tk.Frame(button_and_gallery_frame)
gallery_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)



# Create the gallery canvas with a fixed width
gallery_canvas = tk.Canvas(gallery_frame, width=150)  # Set the width to match gallery_frame width
gallery_scroll_y = tk.Scrollbar(gallery_frame, orient=tk.VERTICAL, command=gallery_canvas.yview)

# Create a frame to hold the gallery items
gallery_frame = tk.Frame(gallery_canvas)

# Add the frame to the canvas
gallery_canvas.create_window((0, 0), window=gallery_frame, anchor=tk.NW)
gallery_frame.bind("<Configure>", lambda e: gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all")))


# Pack the scroll bars and canvas
gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
gallery_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
gallery_canvas.configure(yscrollcommand=gallery_scroll_y.set)

auto_box_checkbutton = tk.Checkbutton(button_frame, text="Auto Box (b)", variable=auto_box_enabled, onvalue=True,
                                      offvalue=False)
auto_box_checkbutton.pack(pady=5)

# Call bind_keys to set up key bindings
bind_keys()

# Bind the resize event to adjust canvas size
root.bind("<Configure>", resize_canvas)

# Bind the mouse click event to canvas
canvas.bind("<Button-1>", on_canvas_click)

root.mainloop()