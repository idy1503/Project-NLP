from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from wordcloud import STOPWORDS, WordCloud
import matplotlib.pyplot as plt

f = open("Curious Myt.txt")

bow = CountVectorizer(stop_words='english')

bow.fit(f)
c = bow.get_feature_names_out()


file = pd.read_table("test.txt")
text = ' '.join(file['Lecteur'].astype(str).tolist())

text = re.sub(r'[^A-Za-z\s]', '', text)

text = text.lower()

stopwords = set(STOPWORDS)
text = ' '.join(word for word in text.split() if word not in stopwords)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title("IMDB Movie Reviews Word Cloud")
plt.show()
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title("IMDB Movie Reviews Word Cloud")
plt.show()
