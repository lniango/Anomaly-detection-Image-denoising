# Autoencoders with Python

A comprehensive implementation of various autoencoder architectures using TensorFlow/Keras, covering shallow autoencoders, deep autoencoders, convolutional autoencoders, and denoising autoencoders. This project demonstrates multiple applications including image reconstruction, dimensionality reduction, anomaly detection, and image denoising.

## 🎯 Project Overview

### What are Autoencoders?

Autoencoders are neural networks designed to learn efficient data representations in an unsupervised manner. They compress input data into a lower-dimensional latent space (encoding) and then reconstruct the original data from this compressed representation (decoding).

**Key Components:**
- **Encoder**: Compresses input to latent representation
- **Latent Space**: Compressed representation (bottleneck)
- **Decoder**: Reconstructs input from latent space

<p align="center">
  <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module5/L2/intro_pic.png" width="65%">
</p>

### Applications Covered

1. **Image Reconstruction**: Compress and reconstruct images with minimal loss
2. **Dimensionality Reduction**: Alternative to PCA for feature extraction
3. **Anomaly Detection**: Identify outliers based on reconstruction error
4. **Image Denoising**: Remove noise from corrupted images
5. **Image Compression**: Reduce image size while preserving quality

## 📊 Datasets Used

### 1. MNIST Dataset
- **Type**: Handwritten digits (0-9)
- **Images**: 70,000 grayscale images
- **Size**: 28×28 pixels
- **Use Cases**: Basic autoencoder training, anomaly detection

### 2. Fashion MNIST Dataset
- **Type**: Fashion products (10 categories)
- **Images**: 70,000 grayscale images
- **Size**: 28×28 pixels
- **Use Cases**: Denoising, image reconstruction

### 3. Olivetti Faces Dataset
- **Type**: Human faces
- **Images**: 400 grayscale images
- **Size**: 64×64 pixels
- **Use Cases**: Deep CNN autoencoders, feature extraction

### 4. Concrete Cracks Dataset
- **Type**: Concrete surface images (cracked vs uncracked)
- **Source**: Kaggle
- **Size**: 227×227 pixels
- **Use Cases**: Advanced CNN autoencoders, damage detection

## 🏗️ Architectures Implemented

### 1. Shallow Autoencoders

**Functional API Implementation:**
```python
# Encoder
input_layer = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_layer)

# Decoder
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
```

**Model Subclassing Implementation:**
```python
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Dense(latent_dim, activation='relu')
        self.decoder = Dense(784, activation='sigmoid')
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

### 2. Deep Autoencoders

Multi-layer architecture for learning complex representations:

```python
# Encoder
x = Dense(128, activation='relu')(input_layer)
x = Dense(64, activation='relu')(x)
encoded = Dense(32, activation='relu')(x)

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(784, activation='sigmoid')(x)
```

### 3. Convolutional Autoencoders (CNN)

Best for image data:

**Encoder:**
```python
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
```

**Decoder:**
```python
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
```

### 4. Denoising Autoencoders

Trained to remove noise from corrupted inputs:

```python
# Add noise to training data
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(0, 1, X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0, 1)

# Train on noisy input, clean output
autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=256)
```

## 📁 Project Structure

```
.
├── Autoencoders.ipynb          # Main notebook with all implementations
├── data/
│   ├── mnist/                  # MNIST dataset
│   ├── fashion_mnist/          # Fashion MNIST dataset
│   ├── olivetti_faces/         # Faces dataset
│   └── concrete_crack_images/  # Concrete cracks dataset
├── models/
│   ├── shallow_autoencoder.h5
│   ├── deep_autoencoder.h5
│   ├── cnn_autoencoder.h5
│   └── denoising_autoencoder.h5
└── README.md
```

## 🚀 Installation

### Requirements

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow>=2.10.0
pip install numpy pandas matplotlib
pip install scikit-learn
pip install pillow
pip install jupyter notebook
```

### Quick Setup

```bash
# Install all dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn pillow jupyter

# Launch Jupyter Notebook
jupyter notebook Autoencoders.ipynb
```

## 💻 Usage

### 1. Basic Autoencoder Training

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define architecture
input_dim = 784
latent_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Compile
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(X_train, X_train, 
                epochs=50, 
                batch_size=256,
                validation_data=(X_test, X_test))
```

### 2. Image Reconstruction

```python
# Encode and decode test images
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)

# Visualize
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
```

### 3. Anomaly Detection

```python
# Train on normal data only (e.g., digit 0)
X_train_normal = X_train[y_train == 0]

# Train autoencoder
autoencoder.fit(X_train_normal, X_train_normal, epochs=50)

# Compute reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Set threshold (e.g., 95th percentile)
threshold = np.percentile(mse, 95)

# Detect anomalies
anomalies = mse > threshold
```

### 4. Image Denoising

```python
# Add noise
noise_factor = 0.5
X_noisy = X_train + noise_factor * np.random.normal(0, 1, X_train.shape)
X_noisy = np.clip(X_noisy, 0, 1)

# Train denoising autoencoder
autoencoder.fit(X_noisy, X_train,  # Input: noisy, Target: clean
                epochs=10,
                batch_size=256,
                validation_data=(X_test_noisy, X_test))

# Denoise new images
denoised = autoencoder.predict(X_test_noisy)
```

### 5. Feature Extraction

```python
# Use encoder for dimensionality reduction
encoder = Model(input_layer, encoded)

# Extract features
features = encoder.predict(X_data)

# Use for downstream tasks (classification, clustering, etc.)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(features)
```

## 📊 Key Concepts

### Loss Functions

**Mean Squared Error (MSE):**
```python
loss = 'mse'  # For continuous outputs
```

**Binary Crossentropy:**
```python
loss = 'binary_crossentropy'  # For binary/normalized data
```

### Latent Dimension Selection

The latent dimension determines compression level:

| Latent Dim | Compression | Quality | Use Case |
|------------|-------------|---------|----------|
| 2-3 | Very high | Low | Visualization |
| 32-64 | High | Medium | General purpose |
| 128-256 | Medium | High | Complex images |
| 512+ | Low | Very high | Minimal loss |

### Training Tips

1. **Normalization**: Scale inputs to [0, 1]
2. **Batch Size**: 256-512 for faster training
3. **Epochs**: 50-100 for convergence
4. **Optimizer**: Adam works well (lr=0.001)
5. **Regularization**: Add dropout or L2 to prevent overfitting

## 🎯 Applications & Results

### 1. MNIST Reconstruction

- **Architecture**: Shallow autoencoder (784 → 32 → 784)
- **Loss**: MSE
- **Result**: High-quality reconstruction with 96x compression

### 2. Fashion MNIST Denoising

- **Noise Level**: Gaussian noise (σ=0.5)
- **Architecture**: CNN autoencoder
- **Result**: Effective noise removal, preserved details

### 3. Anomaly Detection

- **Normal Class**: MNIST digit "0"
- **Anomalies**: Other digits
- **Detection Rate**: >90% accuracy using reconstruction error

### 4. Face Reconstruction

- **Dataset**: Olivetti Faces (64×64)
- **Architecture**: Deep CNN autoencoder
- **Latent Dim**: 128
- **Result**: Clear facial feature preservation

### 5. Concrete Crack Detection

- **Dataset**: 40,000 images (227×227)
- **Architecture**: Deep CNN autoencoder
- **Application**: Infrastructure inspection
- **Result**: Accurate crack pattern learning

## 📈 Performance Metrics

### Reconstruction Quality

```python
# Mean Squared Error
mse = np.mean(np.square(X_test - reconstructions))

# Structural Similarity Index (SSIM)
from skimage.metrics import structural_similarity as ssim
ssim_score = ssim(original, reconstructed)
```

### Compression Ratio

```python
original_size = 28 * 28  # 784
latent_size = 32
compression_ratio = original_size / latent_size  # 24.5x
```

## 🔬 Advanced Topics

### Variational Autoencoders (VAE)

Extension with probabilistic latent space:
- Generate new samples
- Interpolate between images
- More robust latent representations

### Sparse Autoencoders

Add sparsity constraint to encourage selective feature learning:
```python
activity_regularizer=regularizers.l1(10e-5)
```

### Contractive Autoencoders

Penalize sensitivity to small input changes:
- More robust features
- Better generalization

## 🛠️ Exercises Included

### Exercise 1: Fashion MNIST Denoising
Build a denoising autoencoder for Fashion MNIST with different noise levels.

### Exercise 2: Custom Architecture
Design your own deep autoencoder for face reconstruction.

**Challenge**: Achieve <0.01 MSE on test set!

## 📚 Learning Objectives

After completing this notebook, you will be able to:

✅ Understand autoencoder architecture and principles  
✅ Implement autoencoders using Keras Functional API  
✅ Implement autoencoders using Model Subclassing  
✅ Apply autoencoders to real-world problems  
✅ Use autoencoders for dimensionality reduction  
✅ Detect anomalies with reconstruction error  
✅ Denoise images effectively  
✅ Build deep and convolutional autoencoders  
✅ Extract meaningful features from images  

## 💡 Tips & Best Practices

### Preventing Overfitting

```python
# Add dropout
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

# Use early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
```

### Data Augmentation for Denoising

```python
# Multiple noise types
gaussian_noise = np.random.normal(0, 0.1, X.shape)
salt_pepper = np.random.choice([0, 1, X], p=[0.05, 0.05, 0.9])
```

### Visualization

```python
def plot_reconstruction(model, X_test, n=10):
    reconstructed = model.predict(X_test)
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(2, n, n + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

## 🔗 Resources

### Datasets
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Olivetti Faces](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset)
- [Concrete Crack Images - Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

### Papers
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks.
- Vincent, P., et al. (2008). Extracting and Composing Robust Features with Denoising Autoencoders.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

### Tutorials
- [TensorFlow Autoencoders Guide](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [Keras Autoencoders Blog](https://blog.keras.io/building-autoencoders-in-keras.html)

## 🤝 Contributing

Improvements welcome! Areas to explore:
- Variational autoencoders (VAE)
- Adversarial autoencoders (AAE)
- Sequence-to-sequence autoencoders
- Graph autoencoders
- 3D data autoencoders

## 📄 License

This project is based on IBM Skills Network course materials. Educational use permitted with attribution.

## 👥 Authors

**Original Authors:**
- Joseph Santarcangelo (IBM)
- Roxanne Li (IBM Skills Network)

**Contributors:**
- David Pasternak
- Sam Prokopchuk
- Steve Hord

## 🎓 Course Information

This notebook is part of the IBM Developer Skills Network ML311 course on Coursera.

**Estimated Time**: 60 minutes

## 📞 Support

For questions or issues:
1. Review the notebook cells carefully
2. Check TensorFlow documentation
3. Consult the discussion forums
4. Open an issue on GitHub

---

## Quick Start

```bash
# 1. Clone or download the notebook
git clone <your-repo-url>

# 2. Install dependencies
pip install tensorflow numpy matplotlib scikit-learn pillow jupyter

# 3. Launch Jupyter
jupyter notebook Autoencoders.ipynb

# 4. Run all cells (Cell → Run All)
```

**Happy Learning!**

Explore the fascinating world of unsupervised learning with autoencoders!
