import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('punkt')

# Load the data from CSV
data = pd.read_csv('amazon_reviews.csv')
stars = data['Stars'].astype(str).apply(lambda x: x[:3])
stars = stars.astype(float)
data = data[stars < 2]
# Extract the 'Review' column
reviews = data['Review']

# Tokenize the reviews
tokenized_reviews = []
for review in reviews:
    # Check if review is not null and convert it to a string
    if pd.notnull(review):
        review = str(review)
        tokens = nltk.word_tokenize(review)
        tokenized_reviews.append(tokens)

# Convert tokenized reviews back to strings
review_strings = [' '.join(tokens) for tokens in tokenized_reviews]

# Create a CountVectorizer object
vectorizer = CountVectorizer(max_features=1000, lowercase=True, stop_words='english')

# Generate the document-term matrix
dtm = vectorizer.fit_transform(review_strings)

# Create the LDA model
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)

# Fit the LDA model to the document-term matrix
lda_model.fit(dtm)

# Get the most prominent topics
num_words = 10
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda_model.components_):
    top_features_idx = topic.argsort()[:-num_words - 1:-1]
    top_features = [feature_names[i] for i in top_features_idx]
    print(f"Topic #{topic_idx+1}:")
    print(", ".join(top_features))
    print()



