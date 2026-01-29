#step 1:-Library Import
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#Data Load
df=pd.read_csv("movie.csv")
#print(data.head())

'''corrct datatype for year'''
#year me (2019) aesa tha jise remove karke int bana ne kel liye:
df['Year']=df['Year'].astype(str)
df['Year']=df['Year'].str.replace(r'[^0-9]','',regex=True)
df['Year']=pd.to_numeric(df['Year'],errors='coerce')
# Safe way: fill missing with median, phir int
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Year'] = df['Year'].astype(int)

'''corrct datatype for duration'''
df['Duration']=df['Duration'].astype(str)
df['Duration']=df['Duration'].str.replace(r'[^0-9]','',regex=True)
df['Duration']=pd.to_numeric(df['Duration'],errors='coerce')
# Safe way: fill missing with median, phir int
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Duration'] = df['Duration'].astype(int)

'''corrct datatype for votes'''
df['Votes']=df['Votes'].astype(str)
df['Votes']=df['Votes'].str.replace(r'[^0-9]','',regex=True)
df['Votes']=pd.to_numeric(df['Votes'],errors='coerce')
# Safe way: fill missing with median, phir int
df['Votes'] = df['Votes'].fillna(df['Votes'].median())
df['Votes'] = df['Votes'].astype(int)

#print(df.dtypes)
'''Handling the missing value'''
print(df.isnull().sum())
df['Year']=df['Year'].fillna(df['Year'].median())
df['Duration']=df['Duration'].fillna(df['Duration'].median())
df['Genre']=df['Genre'].fillna("Unknown")
df['Rating']=df['Rating'].fillna(df['Rating'].median())
df['Votes']=df['Votes'].fillna(df['Votes'].median())
df['Director']=df['Director'].fillna('Unknown')
df['Actor 1']=df['Actor 1'].fillna('Unknown')
df['Actor 2']=df['Actor 2'].fillna('Unknown')
df['Actor 3']=df['Actor 3'].fillna('Unknown')
print(df.isnull().sum())

'''label encoder create '''
le_genre=LabelEncoder()
le_director=LabelEncoder()
le_actor1=LabelEncoder()
le_actor2=LabelEncoder()
le_actor3=LabelEncoder()

#Text to convert into numeric
df['Genre']=le_genre.fit_transform(df['Genre'])
df['Director']=le_director.fit_transform(df['Director'])
df['Actor 1']=le_actor1.fit_transform(df['Actor 1'])
df['Actor 2']=le_actor2.fit_transform(df['Actor 2'])
df['Actor 3']=le_actor3.fit_transform(df['Actor 3'])

print('text column convert numericx successfully')

'''Model Train'''
#model input and output create
x = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3','Year', 'Duration', 'Votes']]
y = df['Rating']

# Check X types
print(x.dtypes)


#create a model
model=LinearRegression()
model.fit(x,y)

print('model trained successfullly')

'''User input'''
print("\n--- ENTER NEW MOVIE DETAILS ---")

# Text Inputs
movie=input('Enter your movie name:')
genre = input("Enter Genre: ").strip()
director = input("Enter Director: ").strip()
actor1 = input("Enter Actor 1: ").strip()
actor2 = input("Enter Actor 2: ").strip()
actor3 = input("Enter Actor 3: ").strip()

# Numeric Inputs
year = int(input("Enter Year of release: "))
duration = int(input("Enter Duration (minutes): "))
votes = int(input("Enter Votes: "))

#safety tansform input
if genre not in le_genre.classes_:
    print(f"Note: Genre '{genre}' not in training data, default 'Unknown' used.")
    genre = 'Unknown'
if director not in le_director.classes_:
    print(f"Note: Director '{director}' not in training data, default 'Unknown' used.")
    director = 'Unknown'
if actor1 not in le_actor1.classes_:
    actor1 = 'Unknown'
if actor2 not in le_actor2.classes_:
    actor2 = 'Unknown'
if actor3 not in le_actor3.classes_:
    actor3 = 'Unknown'

# Convert to numeric using LabelEncoder
genre=le_genre.transform([genre])[0]
director = le_director.transform([director])[0]
actor1 = le_actor1.transform([actor1])[0]
actor2 = le_actor2.transform([actor2])[0]
actor3 = le_actor3.transform([actor3])[0]

''' prediction output'''
new_movie = [[genre, director, actor1, actor2, actor3, year, duration, votes]]
prediction=model.predict(new_movie)[0]

print(f"Rating of your movie {movie}:",prediction.round())
