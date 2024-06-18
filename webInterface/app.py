from flask import Flask, render_template, request, Response, stream_with_context, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Set random seed for reproducibility
tf.random.set_seed(272)

# Define the image size used by the model
img_size = 400

# Load the VGG19 model without the top layer
vgg_model = tf.keras.applications.VGG19(include_top=False,
                                        input_shape=(img_size, img_size, 3),
                                        weights='imagenet')

# Freeze the model weights
vgg_model.trainable = False

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)
]

def preprocess_image(image_path, img_size):
    image = np.array(Image.open(image_path).resize((img_size, img_size)))
    image = tf.constant(np.reshape(image, ((1,) + image.shape)))
    return image

def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_img(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def gram_matrix(feature_map):
    gram_matrix = tf.matmul(feature_map, tf.transpose(feature_map))
    return gram_matrix

def get_layer_outputs(model, layer_names):
    outputs = [model.get_layer(layer[0]).output for layer in layer_names]
    feature_extractor_model = tf.keras.Model([model.input], outputs)
    return feature_extractor_model

def initialize_generated_image(content_image):
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    return generated_image

def calculate_content_loss(content_output, generated_output):
    content_features = content_output[-1]
    generated_features = generated_output[-1]
    _, height, width, channels = generated_features.get_shape().as_list()
    content_features_unrolled = tf.reshape(content_features, shape=[-1, channels])
    generated_features_unrolled = tf.reshape(generated_features, shape=[-1, channels])
    content_loss = tf.reduce_sum(tf.square(tf.subtract(content_features_unrolled, generated_features_unrolled))) / (4 * height * width * channels)
    return content_loss

def calculate_layer_style_loss(style_features, generated_features):
    _, height, width, channels = generated_features.get_shape().as_list()
    style_features = tf.transpose(style_features, perm=[3, 1, 2, 0])
    generated_features = tf.transpose(generated_features, perm=[3, 1, 2, 0])
    style_features = tf.reshape(style_features, shape=[channels, height*width])
    generated_features = tf.reshape(generated_features, shape=[channels, height*width])
    gram_style = gram_matrix(style_features)
    gram_generated = gram_matrix(generated_features)
    layer_style_loss = tf.reduce_sum(tf.square(tf.subtract(gram_style, gram_generated))) / (4 * channels**2 * (height * width)**2)
    return layer_style_loss

def calculate_style_loss(style_outputs, generated_outputs, STYLE_LAYERS):
    style_loss = 0
    style_features = {layer_name: style_outputs[i] for i, (layer_name, _) in enumerate(STYLE_LAYERS)}
    generated_features = {layer_name: generated_outputs[i] for i, (layer_name, _) in enumerate(STYLE_LAYERS)}
    for layer_name, weight in STYLE_LAYERS:
        style_layer_loss = calculate_layer_style_loss(style_features[layer_name], generated_features[layer_name])
        style_loss += weight * style_layer_loss
    return style_loss

@tf.function()
def total_cost(content_loss, style_loss, alpha=10, beta=40):
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

@tf.function()
def training_function(generated_image, content_features, style_features, STYLE_LAYERS, optimizer, output_vgg):
    with tf.GradientTape() as tape:
        generated_features = output_vgg(generated_image)
        style_loss = calculate_style_loss(style_features, generated_features, STYLE_LAYERS)
        content_loss = calculate_content_loss(content_features, generated_features)
        total_loss = total_cost(content_loss, style_loss, alpha=10, beta=40)
    grad = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_image(generated_image))
    return total_loss

def style_transfer(content_path, style_path, epochs):
    content_image = preprocess_image(content_path, img_size)
    style_image = preprocess_image(style_path, img_size)
    output_vgg = get_layer_outputs(vgg_model, STYLE_LAYERS + [('block5_conv4', 1)])
    content_target = output_vgg(content_image)
    style_target = output_vgg(style_image)
    ppdcontent = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    content_features = output_vgg(ppdcontent)
    ppdstyle = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    style_features = output_vgg(ppdstyle)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    generated_image = initialize_generated_image(content_image)
    generated_image = tf.Variable(generated_image)
    for i in range(epochs):
        total_loss = training_function(generated_image, content_features, style_features, STYLE_LAYERS, optimizer, output_vgg)
        if i % 250 == 0:
            print(f"Epoch {i}, Loss: {total_loss.numpy()}")
            epoch_output_path = os.path.join('static', f'generated_image_epoch_{i}.jpg')
            generated_image_np = tensor_to_img(generated_image)
            generated_image_np.save(epoch_output_path)
            yield f"data: /static/generated_image_epoch_{i}.jpg\nevent: epoch\n\n"

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        if content_file and style_file:
            content_path = os.path.join('uploads', 'content.jpg')
            style_path = os.path.join('uploads', 'style.jpg')
            content_file.save(content_path)
            style_file.save(style_path)
            return redirect(url_for('result'))
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/stream')
def stream():
    content_path = os.path.join('uploads', 'content.jpg')
    style_path = os.path.join('uploads', 'style.jpg')
    return Response(stream_with_context(style_transfer(content_path, style_path, 5001)), content_type='text/event-stream')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

