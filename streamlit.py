import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
# Specify the title and logo for the web page.
st.set_page_config(page_title='Medical Inventory Optimization')
                               
st.set_page_config(page_icon = Image.open('C:/Users/varun/OneDrive/Desktop/Project 1/eth.jpg'))

#Specify the title
st.title('MEDICAL INVENTORY OPTIMIZATION')


# Add a sidebar to the web page. 
st.markdown('---')
# Sidebar Configuration
st.sidebar.image('C:/Users/varun/OneDrive/Desktop/Project 1/eth.jpg', width = 250)
st.sidebar.markdown('Medical Inventory Optimization')
st.sidebar.markdown('Maximizing inventory efficiency')
st.sidebar.markdown('We can predict the medicine quantity in upcoming weeks') 

st.sidebar.markdown('---')
st.sidebar.write('Developed by Varun Kadavergu')

        


def forecast(Week, DrugName):

    with open(r'C:/Users/varun/arima_mod.pkl','rb') as f:
        loaded_model = pickle.load(f)
    
    start = 1
    end = len(Week) 
    prediction = loaded_model.predict(start = start,end = end).rename(DrugName)
    prediction.plot(legend = True)
    
    plt.figure(figsize = (10,6))
    plt.plot(prediction, label='Predicted')
    plt.legend()
    plt.show()
    plt.grid(True)
    st.pyplot()
    
    print(prediction)
    return prediction


def main():
     
     Week = st.slider('Weeks of prediction:', 1, 52, 1)
     DrugName = st.selectbox('Select the drug to forecast', ['LEVOSALBUTAMOL/LEVALBUTEROL 0.63MG RESPULES',
                                                              'MULTIPLE ELECTROLYTES 500ML IVF',
                                                              'NORADRENALINE 2ML INJ',
                                                              'ONDANSETRON 2MG/ML',
                                                              'PANTOPRAZOLE 40MG INJ',
                                                              'PARACETAMOL 1GM IV INJ',
                                                              'SEVOFLURANE 99.97%',
                                                              'SODIUM CHLORIDE 0.9%',
                                                              'SODIUM CHLORIDE IVF 100ML',
                                                              'WATER FOR INJECTION 10ML SOLUTION'])
     
     
        
     #Code for prediction
     result = []
     
     #Creating a button for prediction
     if st.button('Forecast'):
         result =  forecast(range(1, Week+1), DrugName)
         st.success('Forecast for selected weeks : {}'.format(result))    
     
     

if __name__ == '__main__':
     main()
     
