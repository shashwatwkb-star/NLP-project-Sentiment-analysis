# NLP-project-Sentiment-analysis
This project focuses on building a machine learning model to perform sentiment analysis on textual data using Natural Language Processing (NLP) techniques. The objective is to classify movie reviews as positive or negative, enabling automated understanding of user opinions.

The dataset used is the IMDB Movie Reviews Dataset, which contains thousands of labeled reviews. Since the data is unstructured text, extensive preprocessing was required. This included converting text to lowercase, removing HTML tags and special characters, eliminating stopwords, and applying stemming to reduce words to their root forms.

After preprocessing, the text data was transformed into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization. This technique assigns importance to words based on their frequency and relevance across documents, making it suitable for text-based machine learning tasks.

A Naive Bayes classifier was used as the primary model due to its efficiency and effectiveness in handling high-dimensional text data. The dataset was split into training and testing sets (80/20), and the model was evaluated using metrics such as accuracy, precision, recall, and F1-score.

The model achieved strong performance, demonstrating that traditional machine learning approaches combined with proper text preprocessing can yield reliable sentiment classification results. Additionally, the project included data visualization (sentiment distribution) and a custom prediction function to classify new input text.

Overall, this project highlights the importance of preprocessing in NLP, the effectiveness of TF-IDF for feature extraction, and the suitability of probabilistic models like Naive Bayes for text classification tasks. Future improvements could include advanced models such as LSTM or BERT, as well as deployment using a Flask API for real-time predictions.
