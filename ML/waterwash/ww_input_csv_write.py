import numpy
import pandas
import matplotlib.pyplot as plt
from influxdb import DataFrameClient

host = '192.168.4.33'
port = 8086
user = ''
password = ''
db_name = 'Labview'

def data_from_influx():
    data_indb = DataFrameClient(host=host, \
                                           username=user, \
                                           password=password, \
                                           database=db_name)
    data = data_indb.query('SELECT "gas_fuel_flow_x", "ngp", "npt", "t1_temperature", "t5_average_temperature", "turb_air_inlet_filter_dp", "engine_pcd" FROM "Labview"."autogen"."unit1" WHERE time > now() - 5d')

    df = pandas.DataFrame(data['unit1'])
    df.to_csv('~/Desktop/datasets/ww.csv')

data_from_influx()
