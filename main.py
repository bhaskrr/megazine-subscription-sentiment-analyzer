import streamlit as st
import tensorflow as tf

from data_processing import PreprocessAndVectorize

model = tf.keras.models.load_model('./dl_model.keras')
glove_path = './glove.6B.100d.txt'
preprocessor = PreprocessAndVectorize(glove_path)

st.title("Megazine Subscription Sentiment Analyzer")


def print_sample():
    sample_reviews = ["I've been subscribed to this magazine for over a year now, and I absolutely love it! The articles are insightful and well-researched. I look forward to every issue.",
                  "The magazine is overpriced for the quality of content. Most of the articles feel recycled, and nothing new or interesting comes up.",
                  "Subscription is easy, but delivery times are inconsistent. The content is okay, but not worth the wait."]
    st.write("## Sample reviews")
    counter = 1
    for review in sample_reviews:
        st.write(f"**{counter}. {review}**")
        counter += 1
        
def map_sentiment(score):
    if score == 2: return 'Positive'
    elif score == 1: return 'Neutral'
    else: return 'Negative'

with st.form("review_form"):
    user_input = st.text_input("Enter your review:", placeholder="A great mix of topics and keeps me up to date on the latest trends. Highly recommend subscribing!")
    submitted = st.form_submit_button("Submit", type='secondary')
    if submitted:
        processed_text = preprocessor.process(user_input)
        glove_vector = preprocessor.text_to_glove_vector(processed_text)
        glove_vector = glove_vector.reshape(glove_vector.shape[0], 1 ,glove_vector.shape[1])
        prediction = model.predict(glove_vector)
        sentiment = map_sentiment(prediction.argmax())
        st.write(f"'{user_input}'")
        st.write(f"'{sentiment}'")
print_sample()