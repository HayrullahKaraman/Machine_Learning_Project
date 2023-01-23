import streamlit as st
import pandas as pd
import numpy as np


# Page Setting
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="https://roneon.com/wp-content/uploads/2022/12/spam-e-posta-nedir.png",
    menu_items={
        "Get help": "mailto:hayrullahkaraman@gmail.com",
        "About": "For More Information\n" + "https://github.com/HayrullahKaraman"
    }
)

#Info
st.title("Spam Mail Deection")
st.header("About Project")
st.markdown("Mail traffic is increasing day by day, and this mail can have many harmful contents, the mail that comes to you only as a text.Trojans can steal your infection information on your computer with a url in it, or they can open it for a fee, making it unusable.In order not to fall into such traps, Spam Filtering with, prepared by V. Metsis, I. Androutsopoulos and G. Paliouras in 2006 [link](http://www.ceas.cc/)Naive Bayes - Which Naive Bayes?* The Spam mail detection system was prepared based on the compiled and published article.")
st.image("	https://cdn.hosting.com.tr/blog/wp-content/uploads/2020/01/spam-mail-engelleme-3-640x426.jpg")

st.header("Purpose")
st.markdown("It is to classify the content of the incoming mail by using the Natural Language Processing method whether the mail is spam or not.")
st.header("About Dataset")
st.markdown("**Unnammed:0** : Numbering")
st.markdown("**Label** :  **Ham** : Mail is normal, **Spam**: Mail is spam")
st.markdown("**Text** : The content of the mail")
st.markdown("**Label_num** : Value is 0: Mail is normal , Valıue is 1: Mail is spam")
st.header("Simple Dataset")
df=pd.read_csv("spam_ham_dataset.csv")
if st.checkbox("Dataset Show/Hide"):
  st.table(df.sample(3,random_state=42))

st.sidebar.markdown("**Test Your Mail**")
txt = st.sidebar.text_area("Write Your Mail body")
##Model İmport
input_df = pd.DataFrame({
    'text' : [txt]
})




from joblib import load
model=load("smap_mail_lr.pkl")
pred=model.predict([txt])
pred_proba=model.predict_proba([txt])



if st.sidebar.button("Submit"):

    # Info message 
    st.sidebar.info("You can find the result below.")
     
    results_df = pd.DataFrame({
    'Text' : [txt],
    'Prediction':[pred],
    'Confidence' :[pred_proba]
    })
    

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Mail is Normal"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Spam"))
    results_df["Confidence"]=  np.max(pred_proba)*100
    st.sidebar.table(results_df)
else:
    st.sidebar.markdown("Please click the *Submit Button*!")
