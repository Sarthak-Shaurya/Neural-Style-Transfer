<h1>Neural Style Transfer</h1> 
<h2>Introduction</h2>
<h4>Neural Style Transfer (NST) is a captivating technique in computer vision that allows for the creation of novel images by combining one image's content with another's artistic style. Imagine transforming a photograph of your family vacation into a masterpiece reminiscent of Van Gogh's vibrant brushstrokes! This project delves into the world of NST, aiming to construct a system capable of achieving such creative image manipulation.!
</h4>
<h2>Project Overview</h2>
<h4>This code implements a neural style transfer algorithm. It takes a content image and a style image as input and generates an image that combines the content of the content image with the style of the style image. The code first defines a number of helper functions, including functions for preprocessing images, calculating content and style loss, and computing the total loss. Then, it defines a training function that uses a gradient tape to compute gradients and update the generated image iteratively. Finally, it defines a style transfer function that orchestrates the entire style transfer process.</h4>
<h2>Dependencies</h2>
<h4>Flask==2.1.3<br>
tensorflow==2.9.1 <br>
tensorflow-gpu==2.9.1<br>
numpy==1.23.1<br>
Pillow==9.2.0</h4>
<h2>Installation Instructions</h2>
<h4>
  <ol>
    <li> Clone the repo.</li>
    <li> Install the dependencies.</li>
    <li> Create a Flask environment and run the server.</li>
    <li> Go to the server URL.</li>
  </ol>
</h4>
<h2>
  Usage
</h2>
<h4> 


This Flask application performs neural style transfer on images. Here's how to use it:

###  Using the App

1.  **Clone the repository**

2. **Install dependencies**

3. **Run the application:**
python app.py
4. **Access the application**

Open your web browser and visit http://127.0.0.1:5000/ (or localhost:5000 if you're on your local machine).

5. **Upload images:**

* Click "Choose File" and select both a content image and a style image.
* Click "Upload".

6. **View the results**

</h4>
<h2>References</h2>
<h4> •	Coursera Deep Learning Course- Module on CNN- Significant inspiration has been taken from this resource.<br>
•	Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). A neural algorithm of artistic style transfer. arXiv preprint arXiv:1605.08807. https://arxiv.org/abs/1605.08807<br>
•	Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for image counter-fitting. arXiv preprint arXiv:1603.08175. http://arxiv.org/abs/1603.08155<br>
•	Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. https://arxiv.org/pdf/1703.06868.pdf<br>
•	Ghiasi, G. , Lee, H., Kudlur, M. et. al. (2017). Exploring the structure of a real-time, arbitrary neural artistic stylization network. https://arxiv.org/pdf/1705.06830.pdf<br>
</h4>
