from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

pickle_in=open('caloriesPredictor.pkl','rb')
model=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    
    """Let's predict your calories consumption  
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: steps
        in: query
        type: number
        required: true
      - name: exerciseTime
        in: query
        type: number
        required: true
      - name: hoursStood
        in: query
        type: number
        required: true
      - name: summer
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Predicted calories are {}'.format(output))
    

#if __name__=='__main__':
#    app.run(host='0.0.0.0',port=8000)

if __name__=='__main__':
    app.run(debug=True)
