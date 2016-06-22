__author__ = 'Arkenstone'

import pandas as pd
from CLV_paretoNBD_model import *
from connect_db import *

def run_pareto_nbd_model(df, header, k, out_file):
    # variables required
    frequency = np.asarray(df[header[1]])
    age = np.asarray(df[header[3]])
    recency_r = age - np.asarray(df[header[2]])
    # initial the paretoNBD model object
    current_pareto_model = ParetoNBD()
    # training the model parameters using this frequency, age, recency data
    current_pareto_model.model_pars_fit(frequency, recency_r, age)
    # calculate the possibility the customer is still alive now
    p_alive = current_pareto_model.p_alive_present(frequency, recency_r, age)
    # estimate the number of transaction win next k days
    tras_k = current_pareto_model.freq_future(frequency, recency_r, age, k)
    # format a data frame for output
    new_df = pd.DataFrame({'p_alive': p_alive, 'frequency_in_next_k_days': tras_k, 'k': pd.Series([k] * len(p_alive))})
    out_df_frame = [df, new_df]
    # concatenate origin and result data frame
    df_out = pd.concat(out_df_frame, axis=1)
    df_out.to_csv(out_file)

def run_with_db():
    # initial db connect class
    currentDB = extractDataFromDB()
    # database IP, user name, password, database selected
    currentDB.localhost = "xx.xx.xx.xx"
    currentDB.username = "your_user"
    currentDB.password = "your_password"
    currentDB.dbname = "your_db_name"
    currentDB.tbname = "your_table_name"
    currentDB.enterprise_id = "xxx"
    # factors should be retrieved: it should follow the order: customer, frequency, recency, age
    selected = ["customer_id", "frequency", "recency",  "age"]
    # predict transactions made during future k days
    k = 60
    # output file name
    out_file = "test.result.csv"

    # retrieve data from db table: customer behavior
    df = currentDB.get_data_from_db(selected)
    # run the model
    run_pareto_nbd_model(df, selected, k, out_file)

def run_with_file():
    df = pd.DataFrame.from_csv('test.data.csv')
    freq = df.frequency
    rec = df.recency
    age = df.age
    k = 60
    selected = ["customer_id", "frequency", "recency",  "age"]
    out_file = "test.result.csv"
    run_pareto_nbd_model(df, selected, k, outfile)

run_with_db()
run_with_file()
