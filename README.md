![Image](https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/competition/image/93/header_4dd2027d-77d5-415c-aae3-d5ae69b5f9b8.png)

# ZINDI NLP CHALLENGE : [To Vaccinate or Not to Vaccinate](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate)

## Repository description
This repository contains the basic files to start the NLP Live Project based on the Zindi's challenge [To Vaccinate or Not to Vaccinate: It’s not a Question by #ZindiWeekendz](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate).

## Challenge Description
This challenge was designed specifically as a #ZindiWeekendz hackathon (To Vaccinate or Not to Vaccinate: It’s not a Question). We are re-opening the hackathon as a Knowledge Challenge, to allow the Zindi community to learn and test their skills. To help you all out, we’ve created a new Tutorials tab with helpful resources from the community. We encourage Zindians to share their code on the discussion board so that everyone in our community can learn from and support one another.

Work has already begun towards developing a COVID-19 vaccine. From measles to the common flu, vaccines have lowered the risk of illness and death, and have saved countless lives around the world. Unfortunately in some countries, the 'anti-vaxxer' movement has led to lower rates of vaccination and new outbreaks of old diseases.

Although it may be many months before we see COVID-19 vaccines available on a global scale, it is important to monitor public sentiment towards vaccinations now and especially in the future when COVID-19 vaccines are offered to the public. The anti-vaccination sentiment could pose a serious threat to the global efforts to get COVID-19 under control in the long term.

**The objective of this challenge is to develop a machine learning model to assess if a Twitter post related to vaccinations is positive, neutral, or negative. This solution could help governments and other public health actors monitor public sentiment towards COVID-19 vaccinations and help improve public health policy, vaccine communication strategies, and vaccination programs across the world.**


## About
The data comes from tweets collected and classified through Crowdbreaks.org [Muller, Martin M., and Marcel Salathe. "Crowdbreaks: Tracking Health Trends Using Public Social Media Data and Crowdsourcing." Frontiers in public health 7 (2019).]. Tweets have been classified as pro-vaccine (1), neutral (0) or anti-vaccine (-1). The tweets have had usernames and web addresses removed.

The objective of this challenge is to develop a machine learning model to assess if a twitter post that is related to vaccinations is positive, neutral, or negative.

**Variable definition:**

**tweet_id:** Unique identifier of the tweet

**safe_tweet:** Text contained in the tweet. Some sensitive information has been removed like usernames and urls

**label:** Sentiment of the tweet (-1 for negative, 0 for neutral, 1 for positive)

**agreement:** The tweets were labeled by three people. Agreement indicates the percentage of the three reviewers that agreed on the given label. You may use this column in your training, but agreement data will not be shared for the test set.


Files available for download are:

**Train.csv** - Labelled tweets on which to train your model

**Test.csv** - Tweets that you must classify using your trained model

**SampleSubmission.csv** - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the ID must be correct. Values in the 'label' column should range between -1 and 1.

**NLP_Primer_twitter_challenge.ipynb** - is a starter notebook to help you make your first submission on this challenge.

## Modelling  
For this project, we utilized pre-trained models from the Hugging Face library, which is a popular library for Natural Language Processing (NLP) tasks. Specifically, we made use of the following models:

**ROBERTA**: The ROBERTA model, available under the name "Abubakari/finetuned-Sentiment-classfication-ROBERTA-model".

**BERT**: The BERT model, available under the name "Abubakari/finetuned-Sentiment-classfication-BERT-model".

**DISTILBERT**: The DISTILBERT model, available under the name "Abubakari/finetuned-Sentiment-classfication-DISTILBERT-model".

These models have been fine-tuned on a sentiment classification task, which makes them suitable for analyzing sentiment in tweets related to vaccinations. By leveraging these pre-trained models, we can benefit from their contextual understanding of language and their ability to extract meaningful representations from text.

During our modeling process, we loaded the selected model based on the choice made by the user. For example, if the user chose the ROBERTA model, we loaded the corresponding model from the Hugging Face library using the "Abubakari/finetuned-Sentiment-classfication-ROBERTA-model" identifier.

Additionally, we performed parameter tuning and fine-tuning as necessary to optimize the performance of the selected models. This included adjusting hyperparameters, such as learning rate, batch size, and number of training epochs, to achieve the best possible results.

By leveraging the power of these pre-trained models, we aimed to accurately classify tweets into positive, neutral, or negative sentiment categories, contributing to the overall goal of monitoring public sentiment towards COVID-19 vaccinations.

For enhanced performance and faster model training, we utilized a GPU (Graphics Processing Unit) instead of relying solely on the CPU (Central Processing Unit). GPUs are highly efficient in handling parallel computations, making them ideal for accelerating deep learning tasks.
Using Google Colab with free GPU resources allowed us to train and evaluate our models efficiently, even without access to high-end GPU hardware. It provided a convenient and cost-effective solution for leveraging GPU acceleration in our machine learning workflows.

Note that Google Colab sessions have time limits and may disconnect after a period of inactivity. However, you can save your progress and re-establish the connection to the GPU when needed.

## Evaluation
The evaluation metric for this challenge is the **Root Mean Squared Error**.

## Deployment
To deploy the model, follow these steps outlined here 

```bash
https://github.com/aliduabubakari/Covid_vaccine-tweet-analytics-app.git
```
To use the deployed app visit:

```bash
https://huggingface.co/spaces/Abubakari/Sales_Prediction#sales-prediction-app
```
Future Work
[Potential areas for future development and improvement]

## Contact
For any inquiries or questions regarding the project, you can contact:

Name: Alidu Abubakari

Role: Data Analyst

Organization: Azubi Africa


