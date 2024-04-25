import os
import io
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score,confusion_matrix

from models import (read_data,linear_regression,logistic_regression,polynomial_regression,preprocess_data,
                    train_model,evaluate_model,l1_regularization,l2_regularization,translate_text,
                        recommend_courses,summarize_text,generate_study_plan,extract_text_from_pdf,
                        generate_assignments,get_download_link)

st.set_page_config(
    page_title="Teaching-Learning Platform",
    page_icon="👩‍💻",
    layout="wide")
working_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(working_dir)
def automate():
    st.markdown("""
    # Welcome to Automate ML
    
    This application helps you train machine learning models without writing code.
    
    ## How to Use
    
    1. **Select a Dataset:** Choose a dataset from the dropdown menu. 
       - For categorical data: Use classification models (e.g., Logistic Regression, SVM).
       - For continuous data: Use regression models (e.g., Linear Regression, Polynomial Regression).
    
    2. **Choose Features:** Select the features you want to use for training.
    
    3. **Select a Model:** Choose the machine learning model you want to train.
    
    4. **Optional Methods:** You can apply additional methods like Principal Component Analysis or Regularization.
    
    5. **Train the Model:** Click the button to train the selected model.
   
            
""")
    models1 = [ "None","Linear Regression","Logistic Regression","Polynomial Regression","Support Vector Classifier",
           "Random Forest Classifier", "XGBoost Classifier"]
    select = st.selectbox("Select an algorithm to know more about it", models1 )
    col1,col2=st.columns(2)
    if select=="Linear Regression":
        with col1:
            st.markdown("""
                    1. Linear regression is a type of supervised machine learning algorithm that computes the 
                    linear relationship between the dependent variable and one or more independent 
                    features by fitting a linear equation to observed data. 
                    2. In other words, is a data analysis technique that predicts the value of unknown data by using another related and known data value. 
                    It mathematically models the unknown or dependent variable and the known or independent variable 
                    as a linear equation. 
                    3. For instance, suppose that you have data about your expenses and income for last year. 
                    Linear regression techniques analyze this data and determine that your expenses are half your income. 
                    They then calculate an unknown future expense by halving a future known income.
                    4. Linear regression is essentially a way of finding the best-fit line that describes the relationship 
                    between two or more variables. The goal is to use this line to predict future values of the dependent variable, 
                    based on the independent variables.
                    5.The equation for a linear regression line is that of a straight line in coordinate geometry i.e., 
                    y = mx + b, where m is the slope of the line, b is the y-intercept, and x is the independent variable.


                        """)
        with col2:
            st.image("/Users/apple/Downloads/Teaching-Learning platform/linearregression.png", width=500, caption="Linear Regression")
    if select=="Logistic Regression":
        with col1:
            st.markdown("""
                        1. Logistic regression is a data analysis technique that uses mathematics to find the relationships between 
                        two data factors. It then uses this relationship to predict the value of one of those factors based on 
                        the other.The prediction usually has a finite number of outcomes, like yes or no.
                        2. Logistic regression is used for binary classification where we use sigmoid function, that takes input as 
                        independent variables and produces a probability value between 0 and 1.
                        3. For example, let’s say you want to guess if your website visitor will click the checkout button in their 
                        shopping cart or not. Logistic regression analysis looks at past visitor behavior, such as time spent on the 
                        website and the number of items in the cart. It determines that, in the past, if visitors spent more than five 
                        minutes on the site and added more than three items to the cart, they clicked the checkout button. Using this 
                        information, the logistic regression function can then predict the behavior of a new website visitor.   
                        4. Logistic regression predicts the output of a categorical dependent variable. Therefore, the outcome must 
                        be a categorical or discrete value.
                        5. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it
                        gives the probabilistic values which lie between 0 and 1.
                        6. In Logistic regression, instead of fitting a regression line, we fit an “S” shaped logistic function, which 
                        predicts two maximum values (0 or 1
                        """)
        with col2:
            st.image("D:\TY_Semester2\SI\ML_models_streamlit\logisticregression.JPG",width=600,caption="Logistic Regression")
    if select=="Polynomial Regression":
        with col1:
            st.markdown("""
                        1. Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) 
                        and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given as:
                        y= b0+b1x1+ b2x12+ b2x13+...... bnx1n
                        2. Polynomial Regression is a special case of Linear Regression where we fit the polynomial equation on
                        the data with a curvilinear relationship between the dependent and independent variables.
                        3. Polynomial Regression does not require the relationship between the independent and dependent variables
                        to be linear in the data set,This is also one of the main difference between the Linear and Polynomial Regression.
                        """)
        with col2:
            st.image("D:\TY_Semester2\SI\ML_models_streamlit\polynomialregression.JPG",width=500,caption="Simple and Polynomial regression")
    if select=="Support Vector Classifier":
        with col1:
            st.markdown("""
                        1. Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression. 
                        Though we say regression problems as well it’s best suited for classification.
                        2. The main objective of the SVM algorithm is to find the optimal hyperplane in an N-dimensional space that
                        can separate the data points in different classes in the feature space.
                        3. The hyperplane tries that the margin between the closest points of different classes should be as maximum as possible.
                        4. The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line.
                        If the number of input features is three, then the hyperplane becomes a 2-D plane.
                        5. SVM chooses the extreme points/vectors that help in creating the hyperplane. 
                        These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.
                        """)
        with col2:
            st.image("D:\TY_Semester2\SI\ML_models_streamlit\svm.png",width=500,caption="Support Vector Machine")
    if select=="Random Forest Classifier":
        with col1:
            st.markdown("""
                        1. Random Forest algorithm is a powerful tree learning technique in Machine Learning. 
                        2. It works by creating a number of Decision Trees during the training phase. 
                        3. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. 
                        4. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance. 
                        5. In prediction, the algorithm aggregates the results of all trees, either by voting (for classification tasks) or by averaging (for regression tasks) This collaborative decision-making process, supported by multiple trees with their insights, provides an example stable and precise results.
                        """)
        with col2:
            st.image("D:\TY_Semester2\SI\ML_models_streamlit\sample.png", caption="Random Forest")
    if select=="XGBoost Classifier":
        with col1:
            st.markdown("""
                        1. XGBoost is an optimized distributed gradient boosting library designed for efficient and scalable training of machine learning models. 
                        2. It is an ensemble learning method that combines the predictions of multiple weak models to produce a stronger prediction. 
                        3. XGBoost stands for “Extreme Gradient Boosting” and it has become one of the most popular and widely used machine learning algorithms due to its ability to handle large datasets and its ability to achieve state-of-the-art performance in many machine learning tasks such as classification and regression.
                        4. One of the key features of XGBoost is its efficient handling of missing values, which allows it to handle real-world data with missing values without requiring significant pre-processing.
                        5. Additionally, XGBoost has built-in support for parallel processing, making it possible to train models on large datasets in a reasonable amount of time.
                    """)
        with col2:
            st.image("xgboost.png",caption="XGBoost Classifier")

    st.write("Now, try out different models by choosing a dataset below for better a understanding.")
    dataset_list = os.listdir(f"{parent_dir}/data")

    dataset = st.selectbox("Select a dataset from the dropdown",
                        dataset_list,
                        index=None)

    df = read_data(dataset)

    if df is not None:
        st.dataframe(df.head())
        
        col1, col2, col3,col4 = st.columns(4)

        def check_continuous_data(df):
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) == len(df.columns):
                return True
            else:
                return False
            
        def check_categorical_data(df):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) == len(df.columns):
                return True
            else:
                return False

        methods=["None","L1 Regularization", "L2 Regularization"]

        model_list = [ "Linear Regression","Logistic Regression","Support Vector Classifier","Random Forest Classifier",
            "XGBoost Classifier","Polynomial Regression","None"]
        model_dictionary = {
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier()
        }
        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            features = [col for col in df.columns if col != target_column]
            if not features:
                st.warning("Please select at least one feature.")
            scaler_type = st.multiselect("Select features for training", features)
        with col3:
            selected_model = st.selectbox("Select a Model", model_list )
        with col4:
            method_name = st.selectbox("Would you like to perform any of these methods?",methods)

        if selected_model == "Polynomial Regression":
            degree = st.slider("Select the degree of polynomial", 1, 6, 2)
        
        def models():
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

            model_to_be_trained = model_dictionary[selected_model]

            model = train_model(X_train, y_train, model_to_be_trained)

            accuracy,cm = evaluate_model(model, X_test, y_test)

            st.success("Test Accuracy: " + str(accuracy))

            st.subheader("Confusion Matrix")
            # plt.figure(figsize=(3, 2))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            # st.pyplot(plt)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            resized_image = image.resize((600, 500))
            st.image(resized_image)

        if st.button("Train the Model"):
            if selected_model == "Linear Regression":
                if check_categorical_data(df):
                    st.error("Cannot train a model on a dataset with only categorical features.")
                else:
                    linear_regression(df,scaler_type,target_column)
            elif selected_model == "Logistic Regression":
                if check_categorical_data(df):
                    st.error("Cannot train a model on a dataset with only categorical features.")
                else:
                    logistic_regression(df,scaler_type,target_column)
            elif selected_model == "Polynomial Regression":
                if check_categorical_data(df):
                    st.error("Cannot train a model on a dataset with only categorical features.")
                else:
                    polynomial_regression(df,scaler_type,target_column,degree)
            elif selected_model == "Support Vector Classifier":
                if check_continuous_data(df):
                    st.error("Cannot train a model on a dataset with only continuous features.")
                else:
                    models()
            elif selected_model == "Random Forest Classifier":
                if check_continuous_data(df):
                    st.error("Cannot train a model on a dataset with only continuous features.")
                else:
                    models()
            elif selected_model == "XGBoost Classifier":
                if check_continuous_data(df):
                    st.error("Cannot train a model on a dataset with only continuous features.")
                else:
                    models()
            elif method_name=="L1 Regularization":
                l1_regularization(df,scaler_type,target_column)
            elif method_name=="L2 Regularization":
                l2_regularization(df,scaler_type,target_column)
def student_options():
    st.write("Student Options:")
    option = st.selectbox("Select an option:", ["None","Automate ML Practical","Translate Text"])
    if option == "Translate Text":
        text_to_translate = st.text_area("Enter text to translate:")
        target_language = st.selectbox("Select target language:", ["Hindi", "French", "German", "Japanese", "Korean"])
        if st.button("Translate"):
            translated_text = translate_text(text_to_translate, target_language)
            st.subheader("Translated Text:")
            st.write(translated_text)
    elif option == "Course Recommendation":
        student_needs = st.text_input("Enter your needs or interests:")
        if st.button("Recommend Courses"):
            recommended_course = recommend_courses(student_needs)
            st.subheader("Recommended Course:")
            st.write(recommended_course)
    elif option=="Automate ML Practical":
        automate()

def teacher_options():
    st.write("Teacher Options:")
    option = st.selectbox("Select an option:", ["None","Text Summarization", "Generate Study Plan", "Generate Assignments"])
    if option == "Text Summarization":
        text = st.text_area("Enter the text to summarize:")
        # Display the summary immediately after inputting the text
        if st.button("Summarize"):
            summary = summarize_text(text)
            st.subheader("Summary:")
            st.write(summary)
    elif option == "Generate Study Plan":
        num_subjects = st.number_input("Enter the number of subjects: ", min_value=1, max_value=10, step=1)
        subjects = {}
        for i in range(num_subjects):
            subject_name = st.text_input(f"Enter subject name {i+1}: ", key=f"subject_name_{i}")
            num_topics = st.number_input(f"Enter the number of topics for {subject_name}: ", min_value=1, max_value=10, step=1, key=f"num_topics_{i}")
            topics = {}
            for j in range(num_topics):
                topic_name = st.text_input(f"Enter topic {j+1}: ", key=f"topic_name_{i}_{j}")
                duration = st.number_input(f"Enter study duration (in minutes) for {topic_name}: ", key=f"topic_duration_{i}_{j}")
                topics[topic_name] = duration
            subjects[subject_name] = topics

        if st.button("Generate Study Plan"):
            study_plan = generate_study_plan(subjects)
            df = pd.DataFrame(study_plan)
            st.write("\nGenerated Study Plan:")
            st.dataframe(df)
    elif option == "Generate Assignments":
        uploaded_file = st.file_uploader("Upload PDF file:")
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            level=st.selectbox("Select an option:", ["Easy","Medium", "Hard"])
            # assignments = generate_assignments(text,level)
            if st.button("Generate"):
                st.write("Generated Assignments:")
                ques=generate_assignments(text,level)
                st.write(ques)
            # for i, assignment in enumerate(assignments, 1):
            #     st.write(f"{i}. {assignment}")
            # # Add option to download assignments as PDF
            # st.markdown(get_download_link(assignments), unsafe_allow_html=True)

st.title("Teaching-Learning through Generative AI")
user_type = st.selectbox("Select user type:", ["Teacher", "Student"])
if user_type == "Teacher":
    teacher_options()
elif user_type == "Student":
    student_options()