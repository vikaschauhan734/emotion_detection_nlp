from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
lemma = WordNetLemmatizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        text = str(request.form['news'])
        # Text Preprocessing
        words = re.findall("[a-zA-Z]*", text)
        text = " ".join(words)
        # Lowering the text
        text = text.lower()
        # Spliting all the words
        text = text.split()
        # Removing Stop Words
        text = [word for word in text if word not in set(stopwords.words('english'))]
        # Lemmatization
        text = [lemma.lemmatize(word) for word in text]
        # Joining all remaining words
        text = " ".join(text)
        
        vectorized = vectorizer.transform([text])

        prediction = model.predict(vectorized)[0]
        output = ""
        if prediction == 0:
            output="Anger"
        elif prediction == 1:
            output="Fear"
        elif prediction == 2:
            output="Joy"
        elif prediction == 3:
            output="Love"
        elif prediction == 4:
            output="Sadness"
        elif prediction == 5:
            output="Surprise"
        else:
            output="Error Occured! Please use different sentence."
    return render_template('prediction.html',result=output)

if __name__=='__main__':
    app.run(debug=True)
