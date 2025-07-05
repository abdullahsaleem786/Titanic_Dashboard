# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# df=pd.read_csv("titanic.csv")
# df_clean=df.dropna()
# print(df_clean.head())
# print("Shape of this dataset: ",df_clean.shape)
# print("Check Column names and dtypes: \n",df_clean.info())
# print("Check Summary of numeric and  categorical: \n",df_clean.describe())
# print("Missing values: \n",df.isnull().sum())
# print("Duplication: \n",df_clean.drop_duplicates())
# # ✅ 2. Basic Statistics / Understanding
# # Count how many passengers survived and how many didn’t
# passager_survive=df[df['Survived']==1]
# print("Surviver are: \n",passager_survive.head())
# # Calculate the survival rate (overall %)
# survival_rate=(len(passager_survive)/len(df))*100
# print("Survival Rate: ",survival_rate)
# # Count how many passengers in each class (Pclass)
# print("Passanger in Pclass: \n",df['Pclass'].value_counts())

# # Count how many males and females
# print("Male and Female : \n",df['Sex'].value_counts())

# # ✅ 3. Grouped Survival Rates
# # Survival rate by gender (Sex)
# rate_bygender=df.groupby("Sex")['Survived'].mean()
# print("Survival rate by gender (Sex): \n",rate_bygender)

# # Survival rate by passenger class (Pclass)
# rate_byclass=df.groupby("Pclass")['Survived'].mean()
# print("Survival rate by passenger class (Pclass): \n",rate_byclass)

# # Survival rate by port of embarkation (Embarked)
# rate_byembarked=df.groupby("Embarked")['Survived'].mean()
# print("Survival rate by port of embarkation (Embarked): \n",rate_byembarked)


# # # ✅ 4. Visual Explorations (Plots)
# # # Histogram or KDE of Age split by survival
# plt.figure(figsize=(10,8))
# sns.histplot(data=df_clean,x='Age',hue='Survived', kde=True, bins=30, palette=['red','green'])
# plt.title("Histogram or KDE of Age split by survival")
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.show()

# # # Bar plot of Survived by Sex
# sns.barplot(x='Sex', y='Survived', data=df_clean, palette='Set2')
# plt.title("Bar plot of Survived by Sex")
# plt.xlabel("Sex ")
# plt.ylabel("Survived")
# plt.show()


# # # Boxplot of Fare grouped by Pclass
# sns.boxplot(x='Pclass', y='Fare', data=df_clean, palette='Set2')
# plt.title("Boxplot of fare by pclass")
# plt.xlabel("Pclass")
# plt.ylabel("Fare")
# plt.show()

# # # Countplot of Embarked to show boarding port distribution
# sns.countplot(data=df_clean,x='Embarked')
# plt.title("Countplot of Embarked to show boarding port distribution")
# plt.show()

# # # Optional: Heatmap to visualize missing values
# sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
# plt.title("Heatmap")
# plt.show()

# # ✅ 5. Data Cleaning
# # Fill or drop missing values (especially in Age, Embarked, and Cabin)
# df['Age'].fillna(df['Age'].mean(), inplace=True)
# df['Embarked'].fillna('Unknown', inplace=True)   # Or use df['Embarked'].mode()[0]
# df['Cabin'].fillna('Missing', inplace=True)
# print(df[['Age','Embarked','Cabin']].isnull().sum())


# # Drop columns not useful for now (like Ticket or Name)
# print("Drop name and ticket column: ",df.drop('Ticket',axis=1,inplace=True))
# print(df.head())

# # Optionally create new features (e.g. extract deck from Cabin)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# App config
st.set_page_config(layout="wide")
st.title("🚢 Titanic Dashboard")

# Load data
df = pd.read_csv("titanic.csv")

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna('Unknown')
df['Cabin'] = df['Cabin'].fillna('Missing')
df.drop('Ticket', axis=1, inplace=True)

# Sidebar Filters
st.sidebar.header("Filters")
selected_class = st.sidebar.multiselect("Select Passenger Class", df['Pclass'].unique(), default=df['Pclass'].unique())
selected_gender = st.sidebar.multiselect("Select Gender", df['Sex'].unique(), default=df['Sex'].unique())
filtered_df = df[(df['Pclass'].isin(selected_class)) & (df['Sex'].isin(selected_gender))]

# Show data preview
st.subheader("📋 Dataset Preview")
st.dataframe(filtered_df.head())

# Basic Stats
st.subheader("🔎 Basic Statistics")
col1, col2, col3 = st.columns(3)
survived = df['Survived'].sum()
not_survived = len(df) - survived
survival_rate = survived / len(df) * 100
col1.metric("Survived", survived)
col2.metric("Not Survived", not_survived)
col3.metric("Survival Rate", f"{survival_rate:.2f}%")

# Grouped survival stats
st.subheader("📊 Grouped Survival Rates")
col4, col5, col6 = st.columns(3)
with col4:
    st.write("By Gender")
    st.dataframe(df.groupby("Sex")["Survived"].mean())
with col5:
    st.write("By Class")
    st.dataframe(df.groupby("Pclass")["Survived"].mean())
with col6:
    st.write("By Embarked")
    st.dataframe(df.groupby("Embarked")["Survived"].mean())

# 📈 Plots Section
st.subheader("📉 Visual Explorations")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Age Histogram", "Survival by Gender", "Fare by Class", "Embarked Count", "Correlation Heatmap"
])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.histplot(data=filtered_df, x='Age', hue='Survived', kde=True, bins=30, palette='husl', ax=ax1)
    ax1.set_title("Age Distribution by Survival")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Sex', y='Survived', data=filtered_df, palette='Set2', ax=ax2)
    ax2.set_title("Survival Rate by Gender")
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Pclass', y='Fare', data=filtered_df, palette='Set2', ax=ax3)
    ax3.set_title("Fare Distribution by Class")
    st.pyplot(fig3)

with tab4:
    fig4, ax4 = plt.subplots()
    sns.countplot(x='Embarked', data=filtered_df, palette='coolwarm', ax=ax4)
    ax4.set_title("Passenger Count by Embarkation Port")
    st.pyplot(fig4)

with tab5:
    fig5, ax5 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Correlation Heatmap")
    st.pyplot(fig5)

