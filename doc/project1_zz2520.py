
# coding: utf-8

# # What made you happy today?

# summary:Before I did this project, I happened to read an interesting article called[A Data Scientist’s Guide to Happiness](https://medium.freecodecamp.org/a-data-scientists-guide-to-happiness-findings-from-the-happy-experiences-of-10-000-humans-fc02b5c8cbc1).In this article, the author drew a conclution with the same data that Happiness is not conditional on demographics. he uses the emotional intensity of text to weigh the happiness. 
# 
# 

# ![Image of Yaktocat](https://cdn-images-1.medium.com/max/1080/1*s28exzDBF4mGyK27s7birg.png)

# the analysis above claim that there’s little change in the spread of happy experiences across these gender, family, and age demographic groups.In order to aviod the same work.my project follws 3 outline:
# 1. EDA and Wordclouds
# 2. NLP 
# 3. Topic Modelling with LDA
# the purpose of this project is to understand what behaviors make people happy. With the help of HappyDB dataset(a corpus of 100,000 happy moments), my project describes the dataset at begining, and implement the NLP method to topiclize the dataset and use the topics to answer the source of happiness.

# In[107]:


import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt


# ## 1.EDA and Wordclouds

# What are the activities described in a given happy moment?

# In[108]:


rawdata = pd.read_csv("C:\\Users\\Andrew\\Documents\\GitHub\\Fall2018-Proj1-andrewzhouxlxl\\output\\processed_moments2.csv")
all_words = rawdata['text'].str.split(expand=True).unstack().value_counts()


# In[109]:


data = [go.Bar(
            x = all_words.index.values[0:50],
            y = all_words.values[0:50],
            marker= dict(colorscale='Jet',
                         color = all_words.values[0:100]
                        ),
            text='cleaned WC'
    )]

layout = go.Layout(
    title='Top 20 Word frequencies for cleaned words'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# First,project uses the cleaned dataset which generate by the two rmd file in doc. and it's easy to find that "friend", "day", "time" are 3 top words.

# In[110]:


from wordcloud import WordCloud
tw = rawdata[0:]['text'].values



# In[111]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="white")
wc.generate(" ".join(str(v) for v in tw))
plt.title("word cloud – an anecdotal overview of the corpus", fontsize=20)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')


# Word cloud is also an intuitive overview of the corpus. The words "friend" and "day" appear most prominently in HappyDB; "wife" and "husband" occur about equally, and so do "son" and "daughter".However,"night" appears more often than "morning" (3268 vs. 2593 times), and "dog" occur much more often than "cat".

# ## 2.NLP

# after the quick view of the dataset, the dataset still seems imperfect. For example, word "day" and "time" seems to appear too many times and may affect the analysis later. so I decide to use the lemmatization again by pacakage from nltk.

# In[112]:


# # Define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)


# In[113]:


from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# ## 3.Topic Modelling with LDA

# In[114]:


# transform datat into a list
import nltk
nltk.download('wordnet')
df = pd.read_csv("C:\\Users\\Andrew\\Documents\\GitHub\\Fall2018-Proj1-andrewzhouxlxl\\doc\\textdata.csv")
text = df['text'].values.astype('U')
# vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
                                     min_df=3,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)


# In[115]:


#lad method
lda = LatentDirichletAllocation(n_components=10, max_iter=10,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)


# In[116]:


lda.fit(tf)


# In[117]:


n_top_words = 30
print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# with the help of sklearn and LatentDirichletAllocation, I generate 10 topics. Next step is to visualize LDA results. I'am going to use a specialized tool called PyLDAVis

# Iteratively, the algorithm goes through each word and reassigns the word to a topic taking into consideration:
# What’s the probability of the word belonging to a topic
# What’s the probability of the document to be generated by a topic

# In[118]:


import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer, mds='tsne')
panel


# from the visualization above, there are the main things should be considered:
# 1. 10 topics are distrubuted evenly in this project,which indicate the freedom will of people. it means that everyone has their perference and seems distrubute diversely.
# 2. more actually happiness is determined by specific things. in topic 1, the high frequence words("famliy" "husband" " song") indicate a experiences that involve their loved ones. also in topic 5 words like play game enjoy seem to show that lots of people reflect very positively on entertainments.
# 

# #### conclusion: so what acctually make people happy?
# These 10 topics represent the foundations of daily happiness for thousands of people.
# 1. From the analysis above, it seems that famliy means a lot for large portion of people and makes them happy. 
# 2. Time beats money. A happier people prefer to have more time in their lives than more money.
# 3. Money is also important.
# 4. Entertainment also helps a lot.

#  References
# Akari Asai, Sara Evensen, Behzad Golshan, Alon Halevy, Vivian Li. HappyDB: A Corpus of 100,000 Crowdsourced Happy Moments
# Jordan Rohrlich. [A Data Scientist’s Guide to Happiness: Findings From the Happy Experiences of 10,000+ Humans](https://medium.freecodecamp.org/a-data-scientists-guide-to-happiness-findings-from-the-happy-experiences-of-10-000-humans-fc02b5c8cbc1)
# [Complete Guide to Topic Modeling](https://nlpforhackers.io/topic-modeling/)
# 
