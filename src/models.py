import os
import io
# import pickle
# import fitz
# import PyPDF2
import base64
import streamlit as st
import pandas as pd
import seaborn as sns
import pdfplumber
from PIL import Image
import google.generativeai as genai
from transformers import pipeline
from summarizer import Summarizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

working_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(working_dir)

GOOGLE_API_KEY = "AIzaSyDjkyqmEZK5UjNuHerdomNwxCAO5Ist4uo"  
genai.configure(api_key=GOOGLE_API_KEY)

def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df
    
def linear_regression(df,selected_features,target):
    housing=df
    X=housing[selected_features]
    y = housing[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)
  
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Actual vs Predicted Price')  
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    resized_image = image.resize((800, 600))
    st.image(resized_image)
    

def logistic_regression(df,selected_features,target):
    diabetes = df

    X = diabetes[selected_features]
    y = diabetes[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    st.success("Test Accuracy: " + str(accuracy))
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    resized_image = image.resize((600, 500))
    st.image(resized_image)

def polynomial_regression(df,selected_features,target,deg):
    try:
        salary=df
        X = salary[selected_features]
        y = salary[target]
        # st.write("Shape of X:", X.shape)
        # st.write("Shape of y:", y.shape)

        lin_regs= LinearRegression()
        lin_regs.fit(X,y)
        Lpredict=lin_regs.predict(X)
        linear_mse=mean_squared_error(y,Lpredict)**0.5
            
        poly_regs= PolynomialFeatures(degree= deg)
        x_poly= poly_regs.fit_transform(X)
        lin_reg_2 =LinearRegression()
        lin_reg_2.fit(x_poly, y)
        Lpredict=lin_reg_2.predict(x_poly)
        poly_mes=mean_squared_error(y,Lpredict)**0.5
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X,y,color="blue")
        plt.plot(X,lin_regs.predict(X), color="red")
        plt.title("Linear Regression")
        plt.xlabel("Position Levels")
        plt.ylabel("Salary")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        resized_image = image.resize((600, 500))
        st.image(resized_image)

        plt.figure(figsize=(8, 6))
        plt.scatter(X,y,color="blue")
        plt.plot(X, lin_reg_2.predict(poly_regs.fit_transform(X)), color="red")
        plt.title("Polynomial Regression")
        plt.xlabel("Position Levels")
        plt.ylabel("Salary")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        resized_image = image.resize((600, 500))
        st.image(resized_image)

        lin_pred = lin_regs.predict([[6.5]])
        st.write("Linear regression prediction:")
        st.success(lin_pred)

        poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))
        st.write("Polynomial regression prediction:")
        st.success(poly_pred)
    except ValueError as e:
        st.error("Error: " + str(e))

def preprocess_data(df, target_column, scaler_type):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) == 0:
        pass
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])


    if len(categorical_cols) == 0:
        pass
    else:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model):
    # training the selected model
    model.fit(X_train, y_train)
    # saving the trained model
    # with open(f"{parent_dir}/trained_model/{model_name}.pkl", 'wb') as file:
    #     pickle.dump(model, file)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy,cm

def l1_regularization(df,selected_features,target):
    x=df[selected_features]
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1)
    lasso = Lasso().fit(x_train, y_train)
    st.success("Lasso Train Score: " + str(lasso.score(x_train, y_train)))
    st.success("Lasso Test Score: " + str(lasso.score(x_test, y_test)))

def l2_regularization(df,selected_features,target):
    x=df[selected_features]
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1)
    ridge = Ridge().fit(x_train, y_train)
    st.success("Ridge Train Score: " + str(ridge.score(x_train, y_train)))
    st.success("Ridge Test Score: " + str(ridge.score(x_test, y_test)))

def translate_text(text, target_language):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Translate the following in" + target_language + "\n" + text )
    summary=response.text
    return summary

    # # Load mBART-50 model and tokenizer
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    # tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

    # # Map target languages to language codes
    # language_codes = {
    #     "Hindi": "hi_IN",
    #     "Telugu": "te_IN",
    #     "Tamil": "ta_IN",
    #     "Marathi": "mr_IN",
    #     "Bengali": "bn_IN",
    #     "Gujarati": "gu_IN",
    #     "Kannada": "kn_IN",
    #     "Malayalam": "ml_IN",
    #     "Punjabi": "pa_IN"
    # }
    # # Tokenize input text and translate to target language
    # inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    # input_ids = inputs["input_ids"]
    # # Translate text to target language
    # generated_tokens = model.generate(
    #     input_ids=input_ids,
    #     forced_bos_token_id=tokenizer.lang_code_to_id[language_codes[target_language]]
    # )
    # # Decode translated text
    # translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # return translated_text[0]


def recommend_courses(interests, academic_background):
    # Initialize a text generation pipeline
    text_generator = pipeline("text-generation", model="gpt2")

    # Generate a prompt based on the inputs
    prompt = f"Based on your interests in {interests} and academic background in {academic_background}, we recommend the following courses:"

    # Generate course recommendations based on the prompt
    generated_text = text_generator(prompt, max_length=100, num_return_sequences=3)

    # Extract the recommended courses from the generated text
    recommended_courses = [sequence['generated_text'].split(":")[1].strip() for sequence in generated_text]

    return recommended_courses

def summarize_text(text):
    # GOOGLE_API_KEY = "AIzaSyDjkyqmEZK5UjNuHerdomNwxCAO5Ist4uo"  
    # genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Summarize following\n" + text)
    summary=response.text
    return summary

def generate_study_plan(subjects):
    study_plan = []
    for subject, topics in subjects.items():
        for topic, duration in topics.items():
            study_plan.append({"Subject": subject, "Topics": topic, "Duration": duration})
    return study_plan

def extract_text_from_pdf(uploaded_file):
    text = ""
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                   page_text = page.extract_text()
                   text += page_text
        except Exception as e:
            st.error(f"Error: {e}")
    return text
    

def generate_assignments(text,level):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Generate questions of " + level + "from the following\n" + text)
    summary=response.text
    return summary

def get_download_link(assignments):
    pdf_content = "\n".join(assignments)
    b64 = base64.b64encode(pdf_content.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="assignments.pdf">Download Assignments as PDF</a>'
    return href