import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
from fractions import Fraction
import cv2 as cv

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

global img_counter
img_counter = 1


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


def clear_directory(directory_path):
    """
    Clears all files in the specified directory.
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

@app.route("/upload", methods=["POST"])
@nocache
def upload():
    global img_counter
    img_counter = 1
    image_processing.upload()
    # Clear the specified directories before processing the upload
    clear_directory("static/img")
    clear_directory("static/cropped_images_random")
    clear_directory("static/cropped_images/")
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    
    # Memeriksa apakah ada file yang diunggah
    if 'file' not in request.files:
        return render_template("home.html", error="Pilih gambar terlebih dahulu.")

    file = request.files['file']

    # Memeriksa apakah nama file tidak kosong
    if file.filename == '':
        return render_template("home.html", error="Pilih gambar terlebih dahulu.")

    # Memeriksa apakah file yang diunggah adalah gambar (image)
    if not allowed_file(file.filename):
        return render_template("home.html", error="File harus berupa gambar.")

    # Create a PIL Image object from the uploaded file
    img = Image.open(file)
    
    # Convert the image to RGB mode
    img = img.convert('RGB')

    img.save("static/img/img1.jpg")
    copyfile("static/img/img1.jpg", "static/img/img_normal.jpg")
    return render_template("uploaded.html", img_counter=img_counter, file_path="img/img1.jpg")

def allowed_file(filename):
    # Mengecek apakah ekstensi file adalah ekstensi gambar yang diterima (misalnya .jpg, .png, .jpeg)
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    global img_counter
    # Hapus gambar-gambar sebelumnya
    for i in range(1, img_counter + 1):
        img_filename = f"static/img/img{i}.jpg"
        if os.path.exists(img_filename):
            os.remove(img_filename)

    copyfile("static/img/img_normal.jpg", "static/img/img1.jpg")
    img_counter = 1
    image_processing.normal()

    return render_template("uploaded.html", img_counter=img_counter, file_path="img/img1.jpg")



@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.grayscale()

    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.zoomin()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.zoomout()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.move_left()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.move_right()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.move_up()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.move_down()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.brightness_addition()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.brightness_substraction()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.brightness_multiplication()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.brightness_division()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.histogram_equalizer()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.edge_detection()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.blur()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.sharpening()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    global img_counter
    img_filename = f"static/img/img{img_counter}.jpg"
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale(img_filename):
        return render_template("histogram.html", img_counter=img_counter, file_paths=[f"img/img{img_counter}.jpg", "img/grey_histogram.jpg"])
    else:
        return render_template("histogram.html", img_counter=img_counter, file_paths=[f"img/img{img_counter}.jpg", "img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"])


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/cropping_susun", methods=["POST"])
@nocache
def cropping_susun():
    global img_counter
    cropping_columns = int(request.form['cropping_columns'])
    cropping_rows = int(request.form['cropping_rows'])

    image_processing.cropping_susun(cropping_rows, cropping_columns)

    # Get the list of cropped image paths
    image_paths = [f"static/cropped_images/cropped_{i}.jpg" for i in range(cropping_columns * cropping_rows)]
    img_filename = f"img/img{img_counter}.jpg"

    return render_template("cropped.html", img_counter=img_counter, image_paths=image_paths, file_path=img_filename, cropping_columns=cropping_columns, cropping_rows=cropping_rows)

@app.route("/cropping_acak", methods=["POST"])
@nocache
def cropping_acak():
    global img_counter
    cropping_columns_random = int(request.form['cropping_columns_random'])
    cropping_rows_random = int(request.form['cropping_rows_random'])

    image_processing.cropping_acak(cropping_rows_random, cropping_columns_random)

    # Get the list of shuffled cropped image paths
    image_paths = [f"static/cropped_images_random/cropped_random_{i}.jpg" for i in range(cropping_columns_random * cropping_rows_random)]
    img_filename = f"img/img{img_counter}.jpg"

    return render_template("cropped.html", img_counter=img_counter, image_paths=image_paths, file_path=img_filename, cropping_columns=cropping_columns_random, cropping_rows=cropping_rows_random)

@app.route("/image/<int:img_num>")
def get_image(img_num):
    if img_num >= 1 and img_num <= img_counter:
        img_filename = f"static/img/img{img_num}.jpg"
        return send_file(img_filename, as_attachment=True)
    else:
        return "Image not found", 404

@app.route("/restore_history/<int:img_num>")
def restore_history(img_num):
    # Set nilai img_counter sesuai dengan nomor gambar yang ditekan
    global img_counter
    image_processing.restore_history(img_num)
    # Hapus gambar dengan nomor yang lebih besar   
    for i in range(img_num + 1, img_counter + 1):
        img_filename = f"static/img/img{i}.jpg"
        if os.path.exists(img_filename):
            os.remove(img_filename)

    # Update img_filename sesuai dengan img_counter yang baru
    img_counter = img_num
    img_filename = f"img/img{img_counter}.jpg"

    # Redirect ke halaman yang sesuai (misalnya, halaman yang menampilkan gambar terbaru)
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)

@app.route("/identity", methods=["POST"])
@nocache
def identity():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.identity()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)

@app.route("/blur_kernel", methods=["POST"])
@nocache
def blur_kernel():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.blur_kernel()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)

@app.route("/blur_cv_blur", methods=["POST"])
@nocache
def blur_cv_blur():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.blur_cv_blur()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)

@app.route("/gaussian_blur/<int:ksize>", methods=["POST"])
@nocache
def gaussian_blur(ksize):
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.gaussian_blur(ksize)
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/median_blur/<int:ksize>", methods=["POST"])
@nocache
def median_blur(ksize):
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.median_blur(ksize)
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/sharp_kernel", methods=["POST"])
@nocache
def sharp_kernel():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.sharp_kernel()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/bilateral_filter", methods=["POST"])
@nocache
def bilateral_filter():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.bilateral_filter()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/zero_padding", methods=["POST"])
@nocache
def zero_padding():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.zero_padding()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/low_filter_pass", methods=["POST"])
@nocache
def low_filter_pass():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.low_filter_pass()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/high_filter_pass", methods=["POST"])
@nocache
def high_filter_pass():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.high_filter_pass()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)


@app.route("/band_filter_pass", methods=["POST"])
@nocache
def band_filter_pass():
    global img_counter
    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    image_processing.band_filter_pass()
    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)

@app.route("/custom_kernel", methods=["POST"])
@nocache
def custom_kernel():
    global img_counter
    img_filename = f"img/img{img_counter}.jpg"

    # Ambil nilai-nilai matriks kernel dari input HTML dan ubah menjadi matriks NumPy
    kernel = np.array([
        [parse_fraction(request.form['cell_1_1']), parse_fraction(request.form['cell_1_2']), parse_fraction(request.form['cell_1_3'])],
        [parse_fraction(request.form['cell_2_1']), parse_fraction(request.form['cell_2_2']), parse_fraction(request.form['cell_2_3'])],
        [parse_fraction(request.form['cell_3_1']), parse_fraction(request.form['cell_3_2']), parse_fraction(request.form['cell_3_3'])]
    ], dtype=np.float32)

    image_path = f"static/img/img{img_counter}.jpg"
    image = cv.imread(image_path)  # Load the image using OpenCV

    # Apply the custom kernel to the image using filter2D
    customKernelImage = cv.filter2D(image, -1, kernel)

    img_counter += 1
    img_filename = f"img/img{img_counter}.jpg"
    # Save the resulting image using OpenCV
    output_img_path = f"static/img/img{img_counter}.jpg"
    cv.imwrite(output_img_path, customKernelImage)

    return render_template("uploaded.html", img_counter=img_counter, file_path=img_filename)

def parse_fraction(fraction_str):
    try:
        # Pisahkan pembilang dan penyebut, kemudian konversi ke float
        parts = fraction_str.split('/')
        if len(parts) == 2:
            numerator, denominator = map(float, parts)
            # Handle jika penyebut adalah nol
            if denominator == 0:
                return 0.0
            else:
                return numerator / denominator
        elif len(parts) == 1:
            # Jika tidak ada '/' dalam string, coba mengonversi ke float
            return float(fraction_str)
    except ValueError:
        # Tangani jika konversi gagal
        return 0.0
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")


        