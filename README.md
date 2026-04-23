# ImageProcessing
Tests and exploration for basic image filtering made for class at Querétaro's Autonomous University (UAQ by it's signals in spanish) for the Master's course in Computational Systems.

The files were made for better understanding how image processing can facilitate further analysis on biological cell segmentation, and Python was used as recommended in class for use of the OpenCV library.

For this project, the following references were used and are recommended:
- https://numpy.org/devdocs/reference/generated/numpy.ndarray.html
- https://docs.opencv.org (for example: https://docs.opencv.org/3.4/d2/d74/tutorial_js_histogram_equalization.html)

## Dependencies
Included in the "requirements.txt" file:
- numpy
- scipy
- scikit-image
- opencv-python

## Usage
Inside the file "algorithm_base.py" all code is written, but "env_test.py" can be used to verify all dependencies were installed correctly and can be used.
Most code was provided to every student, and only a delimited "Preprocessing block" and a variable called <code>imagen_para_segmentar</code> should be edited, adding or removing any further steps to achieve properly visualizing individual cells as much as possible from any of the images in this repository.

#### Steps
1. Install dependencies (either separately or running <code>pip install -r ...path/to/requirements.txt</code> in a terminal
2. Edit the "Preprocessing block" inside a method called <code>ejecutar()</code> to edit, add or remove steps
3. Run with <code>python3 algorithm_base.py <...path/to/image.tif></code>
