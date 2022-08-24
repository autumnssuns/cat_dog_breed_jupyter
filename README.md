# Cat and dog breeds classifier (Transferred EfficientNetV2B0)

[![Heroku](https://img.shields.io/badge/Deployed-DDD?label=Heroku&labelColor=430098&style=for-the-&logo=heroku&logoColor=white)](https://jupyter-applications.herokuapp.com/)

Jupyter Notebook GUI application using IPython widgets to predict cat and dog breeds.

## Running the classifier GUI

### Environment and Dependencies
*This application has been tested to run successfully on a JupyterLab server*

The following external packages are required:
- tensorflow
- matplotlib
- numpy
- PIL
- IPython
- ipywidgets
- ipyfilechooser

The following built-in packages are required, please install if missing:
- sys
- subprocess
- pathlib
- requests
- io
- imageio
- os

### Running the application

Run all cells in the `runner.ipynb` notebook. The GUI will appear at the end of the notebook.

To begin classifying, upload an image using one of the options:
- **From URL**: Copy the URL of the image to the desinated text area and press `Submit`.
- **Upload an image**: Upload an image from your computer. Your image should be classified immediately after uploading.
- **Use Uploaded Image**: Use an image in the current JupyterLab directory tree. Your image should be classified immediately after uploading.

![image](https://user-images.githubusercontent.com/67458114/183231486-dc0cb8a7-2854-4bcd-a555-f0a6e8504ad2.png)

The result will appear on top of the selections, showing your selected image, and a resized version (required by the machine learning model).

![image](https://user-images.githubusercontent.com/67458114/183231688-f1329751-5127-43d0-928e-dcee34141ad2.png)

You can confirm the result by selecting either `Correct` or `Incorrect` button. If the prediction is incorrect, a dialog will appear for you to enter the correct breed.
**By pressing either button, your image will be collected in the Images folder.**
