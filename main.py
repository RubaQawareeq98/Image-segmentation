import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np

###############
# build the main class

class ImageSegmentation:
    def __init__(self, root):
       
        # the main GUI
        self.root = root
        self.root.title("Image Segmentation Program")
        self.root.geometry("1200x600")

        # Configure the basic style of the GUI
        self.style = ttk.Style()
        self.style.theme_use("clam")  
        self.style.configure("TButton", foreground="white", background="#444444")
        self.style.configure("TFrame", background="#444444")
        self.style.configure("TLabel", foreground="white", background="#444444")
        self.style.map("TButton", background=[("active", "#555555")])

        # add Sidebar section for the buttons
        self.sidebar = ttk.Frame(root, width=200, style="TFrame")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
       
        # add Sidebar section for the buttons
        self.sidebar2 = ttk.Frame(root, width=200, style="TFrame")
        self.sidebar2.pack(side=tk.LEFT, fill=tk.Y)

        # Main content area
        self.main_content = ttk.Frame(root, style="TFrame")
        self.main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Add the ocntrol buttons to the sidebar
        # 1- OPen the image
        self.button1 = ttk.Button(self.sidebar, text="Open Images", command=self.open_images, width=25)
        self.button1.pack(pady=5)
        # Convert input image to gray scale
        self.button2 = ttk.Button(self.sidebar, text="Convert to Gray level", command=self.convert_to_gray, width=25)
        self.button2.pack(pady=5)
        # # point detection filter
        self.button3 = ttk.Button(self.sidebar, text="Point Degmentation filter", command=self.point_detection_filter, width=25)
        self.button3.pack(pady=5)
        # # vertical line detection
        self.button4 = ttk.Button(self.sidebar, text="Vertical Line filter", command=self.vertical_line_detection, width=25)
        self.button4.pack(pady=5)
        # horizontal line detection
        self.button5 = ttk.Button(self.sidebar, text="Horizontal Line filter", command=self.horizontal_line_detection, width=25)
        self.button5.pack(pady=5)
        # # +45 line detection
        self.button6 = ttk.Button(self.sidebar, text="45 degree Line filter", command=self.pos45_line_detection, width=25)
        self.button6.pack(pady=5)
        # -45 line detection
        self.button7 = ttk.Button(self.sidebar, text="-45 degree Line filter", command=self.neg45, width=25)
        self.button7.pack(pady=5)
        # # sobel edge detection
        self.button8 = ttk.Button(self.sidebar, text="Horizonta Sobel filter", command=self.sobel_horz_edge_detection, width=25)
        self.button8.pack(pady=5)
        # # sobel edge detection
        self.button9 = ttk.Button(self.sidebar, text="Vertical Sobel filter", command=self.sobel_vert_edge_detection, width=25)
        self.button9.pack(pady=5)
        # # sobel edge detection
        self.button10 = ttk.Button(self.sidebar, text="+45 Sobel filter", command=self.sobel_45_edge_detection, width=25)
        self.button10.pack(pady=5)
        
        self.button11 = ttk.Button(self.sidebar2, text="-45 Sobel filter", command=self.sobel_neg45_edge_detection, width=25)
        self.button11.pack(pady=5)
        
        self.button20 = ttk.Button(self.sidebar2, text="Sobel combined filter", command=self.sobel, width=25)
        self.button20.pack(pady=5)
        
        self.button12 = ttk.Button(self.sidebar2, text="Laplacian filter", command=self.laplacian_filter, width=25)
        self.button12.pack(pady=5)
        # prewitt edge detection
        self.button13 = ttk.Button(self.sidebar2, text="Prewitt filter", command=self.prewitt_edge_detection, width=25)
        self.button13.pack(pady=5)
        
        self.button14 = ttk.Button(self.sidebar2, text="Laplacian Of Gaussian (LoG) filter"
        , command=self.log_filter, width=25)
        self.button14.pack(pady=5)
       
        self.button15 = ttk.Button(self.sidebar2, text="Zero Crossing filter", command=self.zero_crossing_filter, width=25)
        self.button15.pack(pady=5)
        
        self.button16 = ttk.Button(self.sidebar2, text="Adadaptive Thresholding filter", command=self.adaptive_thresholding, width=25)
        self.button16.pack(pady=5,padx=10)
        
        self.button17 = ttk.Button(self.sidebar2, text="Threshold filter", command=self.threshold, width=25)
        self.button17.pack(pady=5)
       
        self.button18 = ttk.Button(self.sidebar2, text="User Defined filter", command=self.user_defined_filter, width=25)
        self.button18.pack(pady=5)
        
        self.button19 = ttk.Button(self.sidebar2, text="Save Output image", command=self.save, width=25)
        self.button19.pack(pady=5)
       

        # Labels for "Before" and "After"
        self.before_label = ttk.Label(self.main_content)
        self.before_label.pack(pady=5)
        self.before_label.place(x=160,y=70)
       
        self.after_label = ttk.Label(self.main_content, text="Output Image")
        self.after_label.pack(pady=5)
        self.after_label.place(x=550,y=70)
       
       
        # Image labels
        self.before_image_label = ttk.Label(self.main_content)
        self.before_image_label.pack(side=tk.LEFT, padx=5)  # Pack "Before" image to the left

        self.after_image_label = ttk.Label(self.main_content)
        self.after_image_label.pack(side=tk.LEFT, padx=5)  # Pack "After" image to the left
        
        # Threshold Image
        self.thre_image_label = ttk.Label(self.main_content)
        self.thre_image_label.pack(side=tk.LEFT, padx=5)  # Pack "After" image to the left

        
        # User defined filter inputs
        
        
        
         # Keep references to input image
        self.image_path = None
        # ####
        self.modified_img=None
        
        # open image method
            
    def open_images(self):
        # Ask the user to select original image
        self.image_path = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        before_image = Image.open(self.image_path[0])
        before_resized = before_image.resize((400, 400))
        before_photo = ImageTk.PhotoImage(before_resized)
        self.before_image_label.configure(image=before_photo)
        self.before_image_label.image = before_photo  # Keep a reference to avoid garbage collection
        
        
     
        # Update labels
        self.before_label.configure(text="Input Image")
    
    # method to display modified image
    def display_modified_image(self, after_image):
          # Resize the "After" image
            self.modified_img = after_image
            after_resized = after_image.resize((400, 400))
            after_photo = ImageTk.PhotoImage(after_resized)
            self.after_image_label.configure(image=after_photo)
            self.after_image_label.image = after_photo  # Keep a reference to avoid garbage collection
            
    
    # Convert input image to gray level
    def convert_to_gray(self):
        try:
            # Load and convert the image to grayscale
            original_image = Image.open(self.image_path[0])
            gray_image = original_image.convert('L')
            self.display_modified_image(gray_image)
          # Update labels
            self.after_label.configure(text="Output Image: ")
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    # point detection filter
    def point_detection_filter(self,threshold=100):
    # Open the image
        try:
            original_image = Image.open(self.image_path[0])
            gray_image = original_image.convert('L')
            filter_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered_image = gray_image.filter(ImageFilter.Kernel((3, 3), filter_kernel.flatten(), scale=1, offset=0))
            
            self.display_modified_image(filtered_image)
            # self.after_label.configure(text="Point Detection Image: ")
            
            np_array = np.array(filtered_image)

            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(np_array.astype(np.uint8))

            # Convert PIL Image to grayscale
            gray_image = pil_image.convert("L")

            # Apply thresholding
            _, thresholded_image = cv2.threshold(np.array(gray_image), 200, 255, cv2.THRESH_BINARY)
        except:
            tk.messagebox.showinfo("","You should select input image at first")
        

    # Vertical point detection
    
    def vertical_line_detection(self):
    # Open the image
        try:
            original_image = Image.open(self.image_path[0])
            # Convert the image to grayscale
            gray_image = original_image.convert('L')
            filter_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
            filtered_image = gray_image.filter(ImageFilter.Kernel((3, 3), filter_kernel.flatten(), scale=1, offset=0))
            
            self.display_modified_image(filtered_image)
            # self.after_label.configure(text= "Vertical Line Detection")
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def horizontal_line_detection(self):
    # Open the image
        try:
            original_image = Image.open(self.image_path[0])
            # Convert the image to grayscale
            gray_image = original_image.convert('L')
            filter_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
            filtered_image = gray_image.filter(ImageFilter.Kernel((3, 3), filter_kernel.flatten(), scale=1, offset=0))
            
            self.display_modified_image(filtered_image)
            # self.after_label.configure(text= "Horizontal Line Detection")
        except:
            tk.messagebox.showinfo("","You should select input image at first")    
    def pos45_line_detection(self, threshold=50):
    # Open the image
        try:
            original_image = Image.open(self.image_path[0])
        # Convert the image to grayscale
            gray_image = original_image.convert('L')
            # Define the custom vertical line detection kernel
            custom_kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])

            filtered_image = gray_image.filter(ImageFilter.Kernel((3, 3), custom_kernel.flatten(), scale=1, offset=0))
            self.display_modified_image(filtered_image)
            # self.after_label.configure(text= "+45 Line Detection")    
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def neg45(self, threshold=50):
        try:
        # Open the image
            original_image = Image.open(self.image_path[0])

        # Convert the image to grayscale
            gray_image = original_image.convert('L')
            # Define the custom vertical line detection kernel
            custom_kernel = np.array([ [2, -1, -1],[-1, 2, -1],[-1, -1, 2]])

            filtered_image = gray_image.filter(ImageFilter.Kernel((3, 3), custom_kernel.flatten(), scale=1, offset=0))
                
            self.display_modified_image(filtered_image)
            # self.after_label.configure(text= "-45 Line Detection")    
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def sobel_horz_edge_detection(self):
        try:
            original_image = cv2.imread(self.image_path[0], cv2.IMREAD_GRAYSCALE)

            sobelx = cv2.Sobel(src=original_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis

            # Combine horizontal and vertical gradients
            sobel_combined = np.sqrt(sobelx**2)

            # Normalize the result to the range [0, 255]
            sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

            # Convert NumPy array to PIL Image
            sobel_image_pil = Image.fromarray(sobel_normalized.astype(np.uint8))

            self.display_modified_image(sobel_image_pil)
        except:
            tk.messagebox.showinfo("","You should select input image at first")    
        
    def sobel_vert_edge_detection(self):
        try:
            original_image = cv2.imread(self.image_path[0], cv2.IMREAD_GRAYSCALE)

            sobely = cv2.Sobel(src=original_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the Y axis

            # Combine horizontal and vertical gradients
            sobel_combined = np.sqrt(sobely**2)

            # Normalize the result to the range [0, 255]
            sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

            # Convert NumPy array to PIL Image
            sobel_image_pil = Image.fromarray(sobel_normalized.astype(np.uint8))

            self.display_modified_image(sobel_image_pil)
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def sobel_45_edge_detection(self):
        try:
            # Open the image
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            kernel_x = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
            prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
            
            # Combine the results from Sobel x and y
            magnitude = np.sqrt(prewitt_x**2)

            # Scale the magnitude to 8-bit for visualization
            scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))

            img= Image.fromarray(scaled_magnitude, 'L')   
            self.display_modified_image(img) 
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def sobel_neg45_edge_detection(self):
        try:
            # Open the image
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            kernel_x = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            

            prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
            
            # Combine the results from Sobel x and y
            magnitude = np.sqrt(prewitt_x**2)

            # Scale the magnitude to 8-bit for visualization
            scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))

            img= Image.fromarray(scaled_magnitude, 'L')   
            self.display_modified_image(img) 
        except:
            tk.messagebox.showinfo("","You should select input image at first")    
    
    def sobel(self):
        try:
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            sobelx = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
            sobely = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
            
            
            # Combine the results from Sobel x and y
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Scale the magnitude to 8-bit for visualization
            scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))

            img= Image.fromarray(scaled_magnitude, 'L')   
            self.display_modified_image(img) 
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def laplacian_filter(self):
        try:
            img = cv2.imread(self.image_path[0])  

            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            src = cv2.GaussianBlur(RGB_img, (3, 3), 0)
            # converting to gray scale
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            edges = cv2.Laplacian(gray, -1, ksize=3, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
            scaled_filtered_image = np.uint8(edges)
            img= Image.fromarray(scaled_filtered_image, 'L')
            self.display_modified_image(img)
        except:
            tk.messagebox.showinfo("","You should select input image at first")
     
    def prewitt_edge_detection(self):
        try:
        # Open the image
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply Prewitt filter for edge detection
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

            prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
            prewitt_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)

            # Combine the results from Prewitt x and y
            magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)

            # Scale the magnitude to 8-bit for visualization
            scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))

            img= Image.fromarray(scaled_magnitude, 'L')
            self.display_modified_image(img)
        except:
            tk.messagebox.showinfo("","You should select input image at first")

    def log_filter(self):
        try:
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to the grayscale image
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Apply Laplacian filter to the blurred image
            laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

            # Convert NumPy array back to PIL Image
            laplacian_image_pil = Image.fromarray(laplacian_image.astype(np.uint8))

            self.display_modified_image(laplacian_image_pil)
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    
    def zero_crossing_filter(self):
        try:
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply Laplacian filter to the grayscale image
            log_filtered = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian blur

            laplacian_image = cv2.Laplacian(log_filtered, cv2.CV_64F)

            # Find zero-crossings in the Laplacian image
            zero_crossings = np.zeros_like(laplacian_image)
            zero_crossings[laplacian_image > 0] = 255

            # Convert NumPy array back to PIL Image
            zero_crossings_pil = Image.fromarray(zero_crossings.astype(np.uint8))

            # Resize the image
            resized_image = zero_crossings_pil.resize((400, 400))

            # Convert PIL Image to Tkinter PhotoImage
            zero_crossings_photo = ImageTk.PhotoImage(resized_image)

            # Configure the label with the PhotoImage
            self.after_image_label.configure(image=zero_crossings_photo)
            self.after_image_label.image = zero_crossings_photo  # Keep a reference to avoid garbage collection
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def adaptive_thresholding(self):
        try:
            thr =  simpledialog.askinteger("","Enter the threshold value")       
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            
            adaptive_thresholded = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, thr)

            # Convert NumPy array back to PIL Image
            adaptive_thresholded_pil = Image.fromarray(adaptive_thresholded)
            self.display_modified_image(adaptive_thresholded_pil)
        except:
            tk.messagebox.showinfo("","You should select input image at first")
    def threshold(self):
        try:
            thr =  simpledialog.askinteger("","Enter the threshold value")
            original_image = cv2.imread(self.image_path[0])

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply a global thresholding method (e.g., binary thresholding)
            _, thresholded = cv2.threshold(gray_image, thr, 255, cv2.THRESH_BINARY)

            thresholded_pil = Image.fromarray(thresholded)
            
            self.display_modified_image(thresholded_pil)
        except:
            tk.messagebox.showinfo("","You should select input image at first")

    def user_defined_filter(self):
        
        filter_size = simpledialog.askstring("","Enter the size of the filter (format: a*a)")
        try:
            size= filter_size.split('*')
            filter_coefficients = simpledialog.askstring("User-Defined Filter", "Enter filter coefficients (comma-separated):")
            if filter_coefficients:
                
                filter_coefficients = list(map(float, filter_coefficients.split(',')))
                
                size=int(size[0])
                original_image = cv2.imread(self.image_path[0])
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                
                custom_kernel = np.array(filter_coefficients).reshape((size, size))
                filtered_image = cv2.filter2D(gray_image, -1, custom_kernel)

                # Scale the filtered image to 8-bit for visualization
                scaled_filtered_image = np.uint8(filtered_image)

                img= Image.fromarray(scaled_filtered_image, 'L')
                self.display_modified_image(img)
        except:
                tk.messagebox.showinfo("","Enter In the correct Formate")
        
    def save(self):
        # Save the modified image
        try:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            self.modified_img.save(save_path)
            tk.messagebox.showinfo("", "The Image Saved Successfully")
        except:
            print("")    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentation(root)
    root.mainloop()
        
        
