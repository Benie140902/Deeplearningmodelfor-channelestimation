# channel-estimation-dl
# Channel Estimation using Deep Learning

This project implements a deep learning approach for channel estimation and modulation scheme classification in wireless communication systems.

## Project Overview

The system uses a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to estimate channel characteristics and classify modulation schemes. It's designed to handle various modulation types including QAM4, QAM16, QAM256, and QPSK in both Rayleigh and Rician fading channels.

## Features

- Data preprocessing and normalization
- Deep learning model for channel estimation and modulation classification
- Support for multiple modulation schemes
- Model training and saving

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow 2.x

## Installation

1. Clone this repository:
git clone https://github.com/Kaushik-4002  /channel-estimation-dl.git
2. Install the required packages:
pip install numpy pandas matplotlib scikit-learn tensorflow
## Usage

1. Ensure your input data files are in the correct location:
- `msg.csv` for input dataset
- Modulation scheme files: `qam4_raydemodsig.csv`, `qam4_ridemodsig.csv`, `qam16_raydemodsig.csv`, `qam16_ridemodrx.csv`, `qam256_raydemodsig.csv`, `qam256_ridemodsig.csv`, `qpsk_raydemodrx.csv`, `qpsk_ridemodrx.csv`
2. Run the main script:
python main.py
## Model Architecture

The deep learning model consists of:
- Two 1D Convolutional layers (64 and 128 filters)
- One LSTM layer (128 units)
- Two Dense layers (128 units and output layer)
- Dropout layers for regularization

## Data

The project uses several datasets for different modulation schemes:
- QAM4 (Rayleigh and Rician)
- QAM16 (Rayleigh and Rician)
- QAM256 (Rayleigh and Rician)
- QPSK (Rayleigh and Rician)

## Results

The model is trained for 50 epochs and saves the trained model as 'model.h5'. After training, it can predict the modulation scheme for new input data and visualize similarity scores for different modulation schemes.

## Visualization

The script generates a plot showing similarity scores for each modulation scheme across different channels.

## Future Work

- Implement more sophisticated similarity score calculation
- Add more comprehensive error handling and input validation
- Explore hyperparameter tuning for improved performance

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

#code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the datasets
input_dataset = pd.read_csv("/content/msg.csv", header=None).values

file_names = [
    "qam4_raydemodsig.csv",
    "qam4_ridemodsig.csv",
    "qam16_raydemodsig.csv",
    "qam16_ridemodrx.csv",
    "qam256_raydemodsig.csv",
    "qam256_ridemodsig.csv",
    "qpsk_raydemodrx.csv",
    "qpsk_ridemodrx.csv"
]

datasets = []
for file in file_names:
    data = pd.read_csv("/content/" + file, header=None).values
    datasets.append(data)

# Concatenate the target datasets
target_dataset = np.concatenate(datasets, axis=0)

# Create labels based on the modulation scheme
labels = np.array(['qam4_ray'] * datasets[0].shape[0] +
                  ['qam4_ri'] * datasets[1].shape[0] +
                  ['qam16_ray'] * datasets[2].shape[0] +
                  ['qam16_ri'] * datasets[3].shape[0] +
                  ['qam256_ray'] * datasets[4].shape[0] +
                  ['qam256_ri'] * datasets[5].shape[0] +
                  ['qpsk_ray'] * datasets[6].shape[0] +
                  ['qpsk_ri'] * datasets[7].shape[0])

# Convert labels to integer labels
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(integer_labels.reshape(-1, 1))

# Repeat the input_dataset for each label to ensure a consistent number of samples
input_dataset = np.repeat(input_dataset, one_hot_labels.shape[0] // input_dataset.shape[0], axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_dataset, one_hot_labels, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model architecture
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(128, 3, activation='relu'))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=50, batch_size=32, validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test))

# Save the trained model
model.save('model.h5')
# Load the trained model
trained_model = load_model('model.h5')

# Manually enter the input data for each channel
input_data = np.array(list(map(int, input("Enter the input data for each channel (5 values): ").split())))

# Ensure the input data has the correct number of features
if len(input_data) != X_train.shape[1]:
    print("Error: Input data should have 5 values.")
else:
    # Reshape and scale the input data
    input_data = np.array(input_data)[:5].reshape(1, -1)  # Take only the first 5 values
    input_data = scaler.transform(input_data)

    # Use the trained model to predict the modulation scheme
    y_pred = trained_model.predict(input_data.reshape(input_data.shape[0], input_data.shape[1], 1))
    y_pred_class = np.argmax(y_pred, axis=1)

    # Define the modulation schemes for each channel
    modulation_schemes = ['qam4_raydemodsig', 'qam4_ridemodsig', 'qam16_raydemodsig', 'qam16_ridemodrx', 'qam256_raydemodsig', 'qam256_ridemodsig', 'qpsk_raydemodrx', 'qpsk_ridemodrx']

    # Calculate the similarity between the input data and each channel in each modulation scheme
    similarity_scores = np.random.rand(len(modulation_schemes), X_test.shape[1])  # Placeholder for similarity scores

    # Find the index of the maximum similarity score
    best_scheme_idx, best_channel_idx = np.unravel_index(similarity_scores.argmax(), similarity_scores.shape)

    # Get the best modulation scheme and channel
    best_scheme = modulation_schemes[best_scheme_idx]
    best_channel = best_channel_idx + 1  # Assuming channels are 1-indexed

    print(f'Best Modulation Scheme: {best_scheme}, Best Channel: {best_channel}')

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    # Plot the similarity scores for each modulation scheme
    for i, ax in enumerate(axs.flatten()):
        ax.bar(range(X_test.shape[1]), similarity_scores[i])
        ax.set_title(f'similarity rate for {modulation_schemes[i]}')
        ax.set_xlabel('Channel')
        ax.set_ylabel('similarity rate')

    plt.tight_layout()
    plt.show()

