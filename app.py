# import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd

# import modelling libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# function for data training
def get_data_train(original_data):
    # separate dependent and independent variables
    x = original_data.drop(['Type of machine failure'], axis=1)
    y = original_data['Type of machine failure']
    col = list(x.columns)
    
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size = 0.6,
                                                        random_state = 180,
                                                        stratify = original_data['Type of machine failure'] # stratify splitting
                                                        )

    # original data latih
    data_train = pd.DataFrame(x_train, columns=col)
    data_train['Type of machine failure'] = list(y_train)
    
    return data_train
    
# function for bagging knn
def bagging_knn(data_train, new_data):
    knn = [] # for knn dataframe
    
    for i in range(1, 51): # 50 knn with 50 different resample
        temp = data_train.sample(n=len(data_train), random_state=i, replace=True).copy()

        # separate x and y
        x_train = temp.drop(['Type of machine failure'], axis=1).copy()
        y_train = temp['Type of machine failure']

        # do the normalization
        col = ['Rotational speed [rpm]', 'Temperature difference [K]',
               'Power [W]', 'Strain [minNm]']
        scaler = MinMaxScaler()
        x_train_scaler = scaler.fit_transform(x_train[col])
        x_test_scaler = np.reshape(new_data, (1, -1))
        x_test_scaler = scaler.transform(x_test_scaler)

        # data training with normalization
        data_train_scaler = pd.DataFrame(x_train_scaler, columns=col)
        data_train_scaler['Type of machine failure'] = list(y_train)

        # create knn model
        knn_boot = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 5)

        # train and save prediction result
        knn_boot.fit(x_train_scaler, y_train)
        knn_boot_pred = knn_boot.predict(x_test_scaler)

        # save as dataframe with transpose
        if i == 1:
            knn = pd.DataFrame({f'pred-{i}':knn_boot_pred}).T
        else:
            temp2 = pd.DataFrame({f'pred-{i}':knn_boot_pred}).T
            knn = pd.concat([knn, temp2])
    
    # voting the prediction
    knn_boot_pred = []
    for i in range(len(knn.columns)):
        temp = knn[i].value_counts().index[0]
        knn_boot_pred.append(temp)
    
    return knn_boot_pred[0]

def main():
    # configure the page information
    st.set_page_config(
        page_title= 'Bagging KNN by Irfan!',
        page_icon= ':gear:'
    )
    
    # initialize and load the data
    loaded_data = pd.read_csv('Include/mod_df.csv')
    loaded_data = get_data_train(loaded_data)

    # configure the title
    st.header('Type of Machine Failure Predictions with Bagging KNN! :gear:', anchor = False)
    st.divider()

    st.header('Input values', anchor = False)

    v_ros = st.number_input(label = 'Rotational speed [rpm]',
                            min_value = int(loaded_data['Rotational speed [rpm]'].min()),
                            max_value = int(loaded_data['Rotational speed [rpm]'].max()))
    col1, col2 = st.columns([2, 2]) # create two columns

    with col1:
        v_at = st.number_input(label = 'Air temperature [K]',
                               min_value = float(loaded_data['Air temperature [K]'].min()),
                               max_value = float(loaded_data['Air temperature [K]'].max()))
        v_pt = st.number_input(label = 'Process temperature [K]',
                               min_value = float(loaded_data['Process temperature [K]'].min()),
                               max_value = float(loaded_data['Process temperature [K]'].max()))
    
    with col2:
        v_tor = st.number_input(label = 'Torque [Nm]',
                               min_value = float(loaded_data['Torque [Nm]'].min()),
                               max_value = float(loaded_data['Torque [Nm]'].max()))
        v_tow = st.number_input(label = 'Tool wear [min]',
                                min_value = float(loaded_data['Tool wear [min]'].min()),
                                max_value = float(loaded_data['Tool wear [min]'].max()))
    
    # do the prediction
    but_predict = st.button("Predict!", type='primary', key='pred1')
    if but_predict:
        st.divider()

        # initial new variables
        v_td = round(abs(v_at - v_pt), 3)
        v_pwe = round(v_tor * v_ros * 0.10472, 3)
        v_str = round(v_tor * v_tow, 3)
        pred = bagging_knn(loaded_data, [v_ros, v_td, v_pwe, v_str])
        
        # give the prediction information
        if pred == 'TWF':
            st.header('Prediction results: :blue[Tool Wear Failure - TWF]', anchor = False)
            st.write('This damage indicates that a part has wear and tear! The replacement tool is required!')
        elif pred == 'PWF':
            st.header('Prediction results: :orange[Power Failure - PWF]', anchor = False)
            st.write('This damage indicates that the engine power is at abnormal limit! The resulting power must be between 3,500 and 9,000 Watt!')
        elif pred == 'OSF':
            st.header('Prediction results: :red[Overstrain Failure - OSF]', anchor = False)
            st.write('This engine fails due to overstrain! The resulting strain must below 11,000 minNm!')
        elif pred == 'HDF':
            st.header('Prediction results: :violet[Heat Dissipation Failure - HDF]', anchor = False)
            st.write('This engine fails due to an abnormal temperature! A better cooling solution is required!')
    
    # copyright claim
    st.divider()
    st.caption('*Copyright (c) Muhammad Irfan Arisani 2024*')
    
if __name__ == '__main__':
    main()