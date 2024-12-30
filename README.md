# Spam Email Classification

This project demonstrates how to classify emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and a Random Forest Classifier.

## Features
- **Preprocessing**: Cleans and processes email text (removes punctuation, converts to lowercase, stems words, and removes stopwords).
- **Vectorization**: Converts text data into numerical format using CountVectorizer.
- **Model Training**: Uses a Random Forest Classifier for prediction.
- **Prediction**: Classifies new emails as Spam or Ham.

## Requirements
- Python 3.7 or higher
- Libraries:
  - `numpy`
  - `pandas`
  - `nltk`
  - `scikit-learn`

Install required libraries:
```bash
pip install numpy pandas nltk scikit-learn
```

## Dataset
The dataset used for this project:
- **Columns**:
  - `text`: The email content.
  - `label_num`: The label (0 for Ham, 1 for Spam).

Replace `'spam_ham_dataset.csv'` with your dataset file.

## How It Works
1. **Data Preprocessing**:
   - Converts text to lowercase.
   - Removes punctuation.
   - Applies stemming to reduce words to their root forms.
   - Removes stopwords (e.g., "the", "is", "in").

2. **Feature Extraction**:
   - Text is converted to a bag-of-words representation using `CountVectorizer`.

3. **Model Training**:
   - Splits data into training and testing sets.
   - Trains a Random Forest Classifier on the training data.

4. **Email Prediction**:
   - Takes an example email, preprocesses it, and predicts if it's Spam or Ham.

## Usage
1. Load the dataset: 
   ```python
   data = pd.read_csv('spam_ham_dataset.csv')
   ```
2. Run the code to train the model and evaluate accuracy:
   ```python
   cl.score(X_test, y_test)
   ```
3. Predict an email:
   ```python
   prediction = cl.predict(x_email)
   print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")
   ```

## Output
- Prints the model's prediction (Spam or Ham) for a sample email.
- Displays the actual label from the dataset for comparison.

## Notes
- Ensure the dataset is in the correct format before running the notebook.
- The `nltk` library requires downloading stopwords:
  ```python
  nltk.download('stopwords')
  ```

## License
Feel free to use and modify this project for learning purposes.
