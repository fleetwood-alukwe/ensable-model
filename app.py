from flask import Flask,request, url_for, redirect, render_template  ## importing necessary libraries
import pickle  ## pickle for loading model(Diabetes.pkl)
import pandas as pd  ## to convert the input data into a dataframe for giving as a input to the model

app = Flask(__name__)  ## setting up flask name

model = pickle.load(open("ensamble_model.pkl", "rb"))  ##loading model
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')             ## Defining main index route
def home():
    return render_template("index.html")   ## showing index.html as homepage


@app.route('/predict',methods=['POST','GET'])  ## this route will be called when predict button is called
def predict(): 
    #int_features=[float(x) for x in request.form.values()]
    Age = request.form['1'] #age of patient      ## Fetching each input field value one by one
    ALB = request.form['2'] #Abumin 
    ALP = request.form['3'] #Alkaline Phosphatase
    ALT = request.form['4'] #Alkanine Aminotransferase 
    AST = request.form['5'] #Aspartate aminotransferase
    BIL = request.form['6'] #Bilirubin
    CHE = request.form['7'] #Acetylcholinesterase
    CHOL = request.form['8'] #Cholesterol
    CREA = request.form['9'] #Creatinine
    GGT = request.form['10'] #Gamma-Glutamyl Transferase
    PROT = request.form['11'] #Total Protein
 
    row_df = pd.DataFrame([pd.Series([Age,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT])])  ### Creating a dataframe using all the values
    print(row_df)
    row_df2 = scaler.transform(row_df)
    prediction=model.predict_proba(row_df2) ## Predicting the output
    output='{0:.{1}f}'.format(prediction[0][1], 2)    ## Formating output

    if output>str(0.5):
       return render_template('index.html',pred='You are not a Blood Donor. \n Your Probability of having Hepatitis is {}'.format(output)) ## Returning the message for use on the same index.html page
    else:
       return render_template('index.html',pred='You are a Blood Donor.\n Your Probability of having Hepatitis is {}'.format(output)) 


if __name__ == '__main__':
    app.run(debug=True)   