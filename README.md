# oibsip_4

# PROBLEM: 

SPAM/JUNK mail is a type of mail that is sent to a massive number of users at a time, Frequently containing cryptic messages, scams or phising content.

# OBJECTIVE:

Using python and ML design a spam detector and train the model to classify it in spam and non-spam category.

# KEY INSIGHTS:

# 01.Data Preparation and Exploration:

Importing Libraries and Mounting Drive: Imports necessary libraries and mounts Google Drive to access the dataset.

Loading Data: Loads the dataset (spam.csv) into a Pandas DataFrame, specifying columns 'label' and 'text' only.

Renaming and Mapping Labels: Renames columns for easier handling and maps 'ham' to 0 and 'spam' to 1 in a new column 'label_num'.

# 02.Data Visualization:

Plotting Distribution: Visualizes the distribution of spam vs ham messages using a count plot from Seaborn.

# 03.Data Splitting and Vectorization:

Train-Test Split: Splits the dataset into training and testing sets (X_train, X_test, y_train, y_test) using train_test_split.

TF-IDF Vectorization: Converts text data into TF-IDF (Term Frequency-Inverse Document Frequency) features using TfidfVectorizer.

# 04.Logistic Regression Model:

Model Initialization and Training: Initializes a Logistic Regression model (LogisticRegression()), fits it on the training data (X_train_tfidf, y_train), and predicts on the test data (X_test_tfidf).

# 05.Model Evaluation: 

Computes accuracy score and confusion matrix to evaluate the performance of the logistic regression model.

Neural Network Model (using Keras):

# 06.Text Tokenization: 

Tokenizes the text data using Tokenizer from Keras.

Sequence Padding: Pads sequences to ensure uniform input size (X_train_pad, X_test_pad).

# 07.Model Architecture and Training:

Neural Network Definition: Defines a Sequential model with an embedding layer, LSTM layer, and dense output layer.

Model Compilation and Summary: Compiles the model with 'adam' optimizer and 'binary_crossentropy' loss, and displays the model summary.

Model Training: Trains the model on the padded training data (X_train_pad, y_train) for 5 epochs.

# 08.Model Evaluation on Test Data:

Evaluation: Evaluates the trained model on the padded test data (X_test_pad, y_test), calculating the test accuracy.

# CONCLUSION:

The output of the final evaluation shows that the LSTM model achieves a test accuracy of approximately 98.21%, demonstrating its effectiveness in classifying spam messages based on textual content. This pipeline covers data loading, preprocessing, model training, and evaluation, providing a comprehensive approach to spam detection using both traditional machine learning (Logistic Regression) and deep learning (LSTM) techniques.
