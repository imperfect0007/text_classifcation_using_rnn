# Text Classification using Recurrent Neural Networks (RNN)

## Overview
This project demonstrates text classification using a Recurrent Neural Network (RNN). The model is designed to classify Amazon reviews based on sentiment. It utilizes PyTorch for model implementation and TensorFlow for text preprocessing.

## Features
- Data preprocessing (handling missing values, tokenization, stopword removal, and lemmatization)
- Tokenization and sequence padding using TensorFlow
- Implementation of an RNN-based text classifier using PyTorch
- Training and evaluation of the model
- Visualization of training results

## Dataset
The project uses the `amazon_reviews.csv` dataset, which contains customer reviews and sentiment labels.

## Requirements
To run this project, install the following dependencies:
```bash
pip install torch torchvision tensorflow numpy pandas matplotlib seaborn scikit-learn nltk
```

## Model Architecture
The RNN model consists of the following layers:
- **Embedding Layer**: Converts words into dense vectors.
- **Recurrent Layer (LSTM/GRU/RNN)**: Captures sequential dependencies in the text.
- **Dense Layer**: Fully connected layer for classification.
- **Output Layer**: Uses softmax or sigmoid activation for multi-class or binary classification.

## Usage
1. Clone the repository:
   ```bash
   [git clone https://github.com/imperfect0007/text-classification-rnn.git](https://github.com/imperfect0007/text_classifcation_using_rnn)
   cd text-classification-rnn
   ```

2. Run the script to preprocess and train the model:
   

3. Evaluate the model on test data:
  

4. Predict using new input:
 

## Results
The model's performance is evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix

## Future Improvements
- Implementing attention mechanisms for better contextual understanding
- Using pre-trained word embeddings (e.g., Word2Vec, GloVe, BERT)
- Exploring Transformer-based architectures

## License
This project is open-source and available under the MIT License.

## Contributors
Feel free to contribute to the project by submitting issues and pull requests!

