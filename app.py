import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
st.set_option('deprecation.showPyplotGlobalUse', False)

# Specify the title and logo for the web page.
st.set_page_config(page_title='Medical Inventory Optimization',
                   page_icon = 'https://supplementfactoryuk.com/wp-content/uploads/2019/06/xCapsugel-Products-ConiSnap-Hard-A-600x420.jpg.pagespeed.ic.q5ccYLXVW1.webp',
                   layout="wide")


#Specify the title
st.title('MEDICAL INVENTORY OPTIMIZATION')


# Add a sidebar to the web page. 
st.markdown('---')
# Sidebar Configuration
st.sidebar.image('https://lucknow.apollohospitals.com/wp-content/uploads/2021/09/kidney-transplant2.jpg', width=200)
st.sidebar.markdown('Medical Inventory Optimization')
st.sidebar.markdown('Maximizing inventory efficiency')
st.sidebar.markdown('We can predict the quantity of medicine that would be required in following weeks') 

st.sidebar.markdown('---')
st.sidebar.write('Developed by Bhupendra Dewangan')

        
def forecast(Week, DrugName):

    # Load the trained model
    with open(r'C:/Users/varun/arima_mod.pkl','rb') as f:
        loaded_model = pickle.load(f)
    
    # Make predictions for the current column using the trained model
    start = 1
    end = len(Week)
    prediction = loaded_model.predict(start=start,end=end).rename(DrugName)
    plot = prediction.plot(legend=True)
    
    # Plot the actual values and predicted values for the current column
    plt.figure(figsize=(5,3))
    plt.plot(prediction, label='Predicted')
    plt.ylabel("Quantity")
    plt.xlabel("Weeks")
    plt.legend()
    plt.show()
    plt.grid(True)
    st.pyplot()
    
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
