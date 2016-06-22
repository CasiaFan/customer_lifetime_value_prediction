__author__ = "Arkenstone"

import MySQLdb as msdb
import datetime as dt
import pandas as pd
import numpy as np
import random

# class for connect to database
class extractDataFromDB:
    def __init__(self, localhost, username, password, dbname, tbname, enterprise_id):
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.tbname = tbname
        self.enterprise_id = enterprise_id

    def connect_db(self):
        # connect to the database
        db = msdb.connect(host=self.localhost, user=self.username, passwd=self.password, db=self.dbname)
        db_cursor = db.cursor()
        # return a db cursor
        return db_cursor

    def get_data_from_db(self, selected, filter=None):
        ############### NOTE: filter should be list format: ["create_time < '2016-06-02'", "enterprise_id = 256"]
        # and selected should be list like ['customer_id', 'create_time']
        db_cursor = self.connect_db()
        # choose table
        tbname = self.tbname
        # choose enterprise
        enterprise_id = self.enterprise_id
        # choose items
        outID = selected
        selected = ', '.join(selected)
        # filter conditions if exist
        if filter:
            cond = ' and '.join(filter)
            # sql filtering command
            sql = "SELECT " + selected + " FROM " + tbname + " WHERE enterprise_id = " + str(enterprise_id) + " and " + cond
        else:
            sql = "SELECT " + selected + " FROM " + tbname + " WHERE enterprise_id = " + str(enterprise_id)
        # initial a dictionary for holding the data
        my_data = {}
        try:
            # fetch all data selected
            db_cursor.execute(sql)
            results = db_cursor.fetchall()
            count = 0
            for row in results:
                my_data[count] = row
                count += 1
        except:
            print "Error: cannot fetch data from %s" %(tbname)
        # convert the data in dictionary ro data frame
        df = pd.DataFrame.from_dict(my_data, orient='index')
        df.columns = outID
        return df
