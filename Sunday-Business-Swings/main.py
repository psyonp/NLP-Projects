from transformers import pipeline
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data/tweet_data.csv")

#Collect datasets and reverse chronological order (make oldest to newest)
forbes = data["Forbes"].tolist()
forbes.reverse()

business_insider = data["Business Insider"].tolist()
business_insider.reverse()

yahoo_finance = data["Yahoo Finance"].tolist()
yahoo_finance.reverse()

#Save model results
classifier = pipeline("sentiment-analysis")
model_forbes = classifier(forbes)
model_business_insider = classifier(business_insider)
model_yahoo_finance = classifier(yahoo_finance)

score_forbes = []
score_business_insider = []
score_yahoo_finance = []

#Isolate model scores and quantify between -1 and 1
for x in model_forbes:
    if x['label'] == "NEGATIVE":
        x['score'] *= -1
    score_forbes.append(x['score'])
    

for x in model_business_insider:
    if x['label'] == "NEGATIVE":
        x['score'] *= -1
    score_business_insider.append(x['score'])

for x in model_yahoo_finance:
    if x['label'] == "NEGATIVE":
        x['score'] *= -1
    score_yahoo_finance.append(x['score'])

#Visualise data
fig, ax = plt.subplots(3)
fig.suptitle("Sentiment Swings on Most Recent 70 Tweets (oldest to newest)")

ax[0].set_title("Forbes")
ax[0].plot(score_forbes)

ax[1].set_title("Business Insider")
ax[1].plot(score_business_insider,'tab:orange')

ax[2].set_title("Yahoo Finance")
ax[2].plot(score_yahoo_finance, 'tab:green')

plt.show()