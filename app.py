import streamlit as st
import pandas as pd
import pickle
import sklearn
import tensorflow as tf
import numpy as np
from feature_engine.outliers import Winsorizer
import time

def main():
# Give the Name of the Application
    st.title('Hotel Reservations Predictions')

    # import model
    model = pickle.load(open('models.pkl','rb'))


    # Create Submit Form
    with st.form(key='form_parameters'):
        no_of_adults = st.slider('no_of_adults', min_value=0, step=1, max_value=4)
        no_of_children = st.number_input('no_of_children', min_value=0, step=1,max_value=10)
        no_of_weekend_nights = st.number_input('no_of_weekend_nights', min_value=0, step=1,max_value=7)
        no_of_week_nights = st.number_input('no_of_week_nights', min_value=0, step=1,max_value=17)
        type_of_meal_plan = st.sidebar.selectbox(label='type_of_meal_plan', options=['Meal Plan 1','Meal Plan 2','Meal Plan 3','Not Selected'])
        required_car_parking_space = st.sidebar.selectbox(label='required_car_parking_space', options=[0,1])
        room_type_reserved = st.sidebar.selectbox(label='room_type_reserved', options=['Room_Type 1','Room_Type 2','Room_Type 3','Room_Type 4','Room_Type 5','Room_Type 6','Room_Type 7'])
        lead_time = st.number_input('lead_time', min_value=0, step=1,value=57, max_value=400)
        arrival_year = st.sidebar.selectbox(label='arrival_year', options=[2017, 2018])
        arrival_month = st.sidebar.selectbox(label='arrival_month', options=[1,2,3,4,5,6,7,8,9,10,11,12])
        arrival_date = st.sidebar.selectbox(label='arrival_date', options=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
        market_segment_type = st.sidebar.selectbox(label='market_segment_type', options=['Online', 'Offline','Corporate','Complementary','Aviation'])
        repeated_guest = st.sidebar.selectbox(label='repeated_guest', options=[0,1])
        no_of_previous_cancellations = st.number_input('no_of_previous_cancellations', min_value=0, step=1,max_value=100)
        no_of_previous_bookings_not_canceled = st.number_input('no_of_previous_bookings_not_canceled', min_value=0, step=1,max_value=100)
        avg_price_per_room = st.number_input('avg_price_per_room', min_value=0.0, step=0.5,value= 99.0, max_value=800.0)
        no_of_special_requests = st.slider('no_of_special_requests', min_value=0, step=1, max_value=10)

        submitted = st.form_submit_button('Predict')


    # convert into dataframe
    data = pd.DataFrame({ 'no_of_adults':[no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights':[no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [type_of_meal_plan],
        'required_car_parking_space':[required_car_parking_space],
        'room_type_reserved':[room_type_reserved],
        'lead_time':[lead_time],
        'arrival_year':[arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date':[arrival_date],
        'market_segment_type':[market_segment_type],
        'repeated_guest':[repeated_guest],
        'no_of_previous_cancellations':[no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled':[no_of_previous_bookings_not_canceled],
        'avg_price_per_room':[avg_price_per_room],
        'no_of_special_requests':[no_of_special_requests]})


    # model predict
    pred = model.predict(data).tolist()[0]


    if submitted:
            # --- showing progress bar ---
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i+1)
        
            st.success('Prediction Result')
            st.write('Hotel Reservation Prediction is : ')
            if pred == 1:
                st.text('Canceled')
            else:
                st.text('Not Canceled')


    data
    
if __name__ == '__main__':
    main()