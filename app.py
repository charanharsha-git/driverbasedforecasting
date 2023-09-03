from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import save_model, load_model
import os



app = Flask(__name__)

#with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_models_1.pkl', 'rb') as file:
  #model = pickle.load(file)
def model_load(store_nbr):
    model = load_model(f'lstm_models/store_{store_nbr}_model.h5', compile=False)
    return model
with open('pickle_files/time_cov.pkl', 'rb') as file:
  time_cov = pickle.load(file)
with open('pickle_files/list_of_store_specific.pkl', 'rb') as file:
  list_of_store_specific = pickle.load(file)
with open('pickle_files/list_of_train_data.pkl', 'rb') as file:
  list_of_train_data = pickle.load(file)
with open('pickle_files/list_of_test_data.pkl', 'rb') as file:
  list_of_test_data = pickle.load(file)
with open('pickle_files/list_of_train.pkl', 'rb') as file:
  list_of_train = pickle.load(file)
with open('pickle_files/list_of_test.pkl', 'rb') as file:
  list_of_test = pickle.load(file)
with open('pickle_files/list_of_final_df.pkl', 'rb') as file:
  list_of_final_df = pickle.load(file)
with open('pickle_files/list_of_scaler.pkl', 'rb') as file:
  list_of_scaler = pickle.load(file)

def prediction_fn(store_nbr,n_steps,promotion_weightage,promotion_strategy):
  scaler=list_of_scaler[store_nbr-1]
  train_data=list_of_train[store_nbr-1]
  test_data=list_of_test[store_nbr-1]
  target_columns = range(33)
  train_data_scaled = list_of_train_data[store_nbr-1]
  test_data_scaled = list_of_test_data[store_nbr-1]
  final_df=list_of_final_df[store_nbr-1]
  train_ratio = 0.95
  train_size = int(train_ratio * len(final_df))
  def create_sequences(data, sequence_length):
    X = []
    y = []
    for k in range(len(data) - sequence_length):
        X.append(data[k:k+sequence_length,-45:])
        y.append(data[k+sequence_length, target_columns])
    return np.array(X), np.array(y)
  sequence_length=10
  X_train, y_train=create_sequences(train_data_scaled, sequence_length)
  X_test, y_test = create_sequences(test_data_scaled, sequence_length)
  global remaining_data
  if promotion_strategy=='Increase':
      remaining_data = final_df.iloc[train_size:, :]*(1+promotion_weightage/100)
  elif promotion_strategy=='Decrease':
      remaining_data = final_df.iloc[train_size:, :] * (1 - promotion_weightage / 100)
  remaining_data_scaled = scaler.transform(remaining_data)
  X_future, y_future = create_sequences(remaining_data_scaled, sequence_length)
  model=model_load(store_nbr)
  predictions = model.predict(X_future)
  predictions_df=pd.DataFrame(columns=final_df.columns)
  for i in range(0,len(predictions_df.columns)):
    if i<33:
      predictions_df.iloc[:,i]=pd.DataFrame(predictions)[i]
    else:
      predictions_df.iloc[:,i]=pd.DataFrame(test_data_scaled).iloc[-75:,i].reset_index(drop=True)
  predictions_rescaled = scaler.inverse_transform(predictions_df)
  rescaled_predictions=pd.DataFrame(predictions_rescaled).iloc[:,:33]
  rescaled_predictions.columns=test_data.iloc[-75:,:33].columns
  rescaled_predictions=rescaled_predictions.set_index(test_data.iloc[-75:,:33].index)
  actuals=test_data.iloc[-75:,:33]
  mape_df=pd.DataFrame()
  mape=[]
  cols=rescaled_predictions.columns
  for i in cols:
    actual_values=actuals[i]
    predicted_values=rescaled_predictions[i]
    mape_val = np.nanmean(np.abs((actual_values - predicted_values) / np.where((actual_values != 0) & (predicted_values != 0), actual_values, np.nan))) * 100
    mape.append(mape_val)
    print(f"MAPE value for {i} is: {mape_val}%")
  mape_df['Product Name']=cols
  mape_df['MAPE']=mape
  return train_data,actuals,rescaled_predictions.iloc[:n_steps,:]



"""def plotting(store_nbr,n_steps,promotion_weightage, product_name):
    delete_png("static/prediction1.png")
    train,actuals,prediction1=prediction_fn(store_nbr, n_steps, promotion_weightage)
    # Calculate the upper and lower bounds for the interval
    interval = prediction1['sales_'+product_name] * (14.84 / 100)

    # Plotting the train data
    #plt.plot(train.index, train['sales_'+product_name].values, label='Train Data')

    # Plotting the forecasted data
    plt.plot(prediction1.index, prediction1['sales_'+product_name].values, label='Forecasted Data')

    # Plotting the interval
    plt.fill_between(prediction1.index, prediction1['sales_'+product_name].values - interval,
                     prediction1['sales_'+product_name].values + interval,
                     color='gray', alpha=0.3, label='Interval')

    # Adding labels and title to the plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Forecasted Data with Interval')

    # Adding legend
    plt.legend()
    #plt.savefig("static/prediction1.png")
    #image_path="static/prediction1.png"
    return None"""




@app.route('/', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        store_no = int(request.form['storeNo'])
        forecast_days = int(request.form['forecastDays'])
        promotion1 = int(request.form['promotion1'])
        promotion2 = int(request.form['promotion2'])
        promotion3 = int(request.form['promotion3'])
        prom1_stg=request.form.get('rdbt1')
        prom2_stg=request.form.get('rdbt2')
        prom3_stg=request.form.get('rdbt3')

        # product_name = request.form.getlist('productFamily')
        train, actuals, prediction1 = prediction_fn(store_no, forecast_days, promotion1,prom1_stg)
        train, actuals, prediction2 = prediction_fn(store_no, forecast_days, promotion2,prom2_stg)
        train, actuals, prediction3 = prediction_fn(store_no, forecast_days, promotion3,prom3_stg)
        pred_cols = prediction1.columns.tolist()
        #print(prediction1.index)
        print(train.columns)

        # Pass the dataframe to the template for rendering
        return render_template('plot.html', dataframe1=prediction1,dataframe2=prediction2, dataframe3=prediction3,columns=pred_cols,idx=prediction1.index.tolist(),pr_list=[promotion1,promotion2,promotion3],pr_stg_list=[prom1_stg,prom2_stg,prom3_stg],store_no=store_no)

        # Render the initial form
    return render_template('index2.html')
if __name__ == '__main__':
    app.run(debug=True)
