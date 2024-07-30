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

The model is trained for 50 epochs and saves the trained model as 'model.h5'.

## Future Work

- Implement prediction functionality for new data
- Add visualization of training progress and results
- Explore hyperparameter tuning for improved performance

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
