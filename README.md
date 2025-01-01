
# Fake News Detection Using Machine Learning

## Overview
This project implements a machine learning model to detect fake news articles using Natural Language Processing (NLP) techniques. The model uses a Multinomial Naive Bayes classifier with TF-IDF features to classify news articles as either 'fake' or 'true'.

## Performance Metrics
- Accuracy: 98.67%
- Precision and Recall for both classes (fake/true) > 98%
- Well-balanced performance across both categories

## Dataset
Dataset Source: [Fake News Detection Dataset on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
#
This dataset contains 44,898 news articles with the following distribution:
- Fake news articles: 23,481
- True news articles: 21,417


## Technical Implementation
- **Text Preprocessing**: 
  - Lowercase conversion
  - Punctuation removal
  - Text cleaning and normalization

- **Feature Engineering**:
  - TF-IDF Vectorization
  - Maximum 5000 features
  - Unigram-based text representation

- **Model**:
  - Algorithm: Multinomial Naive Bayes
  - Training/Test Split: 80/20
  - Hyperparameters: alpha=1.0 (Laplace smoothing)

## Project Structure
```
fake-news-detection/
├── Fakenews.ipynb  
├── README.md
```

## Requirements
- Python 3.x
- pandas
- scikit-learn
- nltk
- numpy

## Installation
1. Clone the repository
```bash
git clone https://github.com/MGK-VENKATESH/Fake-News-Detection.git
cd Fake-News-Detection
```

2. Install required packages
```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/Fakenews.ipynb
```

2. Follow the notebook cells to:
   - Load and preprocess the data
   - Train the model
   - Evaluate the results

## Future Improvements
- Implement more advanced NLP techniques
- Try different ML algorithms (LSTM, BERT)
- Add feature importance analysis
- Create a web interface for real-time predictions
- Expand the dataset with more recent articles

## Acknowledgments
- Dataset provided by [Bhavik Jikadara on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)

