import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.image("https://img.icons8.com/color/96/news.png", width=70) 
st.title("📰 Fake News Detector")
st.caption("Developed by **Anurag Dubey**")

st.write("Enter a News Article below to check whether it is Fake or Real.")


news_input = st.text_area("📝 Enter News Article:", "")

if st.button("🔍 Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("❌ The News is Fake!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.markdown("✔️ **Fake News Detection Project** | Created **Anurag Dubey**")
