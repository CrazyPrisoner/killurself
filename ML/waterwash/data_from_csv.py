import numpy
import pandas
import matplotlib.pyplot as plt
from influxdb import DataFrameClient

def read_data():
    data = pandas.read_csv('~/Desktop/datasets/ww.csv')
    return data

def info_data():
    data = read_data()
    data.dropna(inplace=True)
    print(data.head(10))
    print(data.info())
    print(data.describe())
    print(data.corr())

    empty_df = pandas.DataFrame(columns=['engine_pcd', 'ngp', 'npt', 't1_temperature', 't5_average_temperature', 'turb_air_inlet_filter_dp','gas_fuel_flow_x'])

    #empty_df.gas_fuel_flow_x=numpy.array(data['gas_fuel_flow_x'].rolling(window=400).mean())
    empty_df.gas_fuel_flow_x = data.gas_fuel_flow_x
    plt.plot(empty_df.gas_fuel_flow_x, 'red');
    plt.title("Gas Flow trend");
    plt.show()

    #empty_df.ngp=numpy.array(data['ngp'].rolling(window=400).mean())
    empty_df.ngp = data.ngp
    plt.plot(empty_df.ngp, 'green');
    plt.title("NGP trend");
    plt.show()


    #empty_df.npt=numpy.array(data['npt'].rolling(window=400).mean())
    empty_df.npt = data.npt
    plt.plot(empty_df.npt, 'blue');
    plt.title("NPC trend");
    plt.show()

    #empty_df.t5_average_temperature=numpy.array(data['t5_average_temperature'].rolling(window=600).mean())
    empty_df.t5_average_temperature = data.t5_average_temperature
    plt.plot(empty_df.t5_average_temperature, 'pink');
    plt.title("T5 temperature trend");
    plt.show()

    #empty_df.turb_air_inlet_filter_dp=numpy.array(data['turb_air_inlet_filter_dp'].rolling(window=400).mean())
    empty_df.turb_air_inlet_filter_dp = data.turb_air_inlet_filter_dp
    plt.plot(empty_df.turb_air_inlet_filter_dp, 'pink');
    plt.title("Turb air ft trend");
    plt.show()

    #empty_df.engine_pcd=numpy.array(data['engine_pcd'].rolling(window=400).mean())
    empty_df.engine_pcd = data.engine_pcd
    plt.plot(empty_df.engine_pcd, 'pink');
    plt.title("Engine pcd trend");
    plt.show()

    #empty_df.t1_temperature=numpy.array(data['t1_temperature'].rolling(window=400).mean())
    empty_df.t1_temperature = data.t1_temperature
    plt.plot(empty_df.t1_temperature, 'pink');
    plt.title("T1 temperature trend");
    plt.show()

    return empty_df

def input_data():
    df = info_data() # values
    divide = int(len(df)*0.8) # value for divide data on train and test
    df.dropna(inplace=True)
    print(df)
    print(df.head(10),"\n") # print first 10 raws
    print(df.info(),"\n") # print info about dataframe
    print(df.shape,"\n") # print dataframe shape
    print(df.describe(),"\n") # print info about values
    print(df.corr(),"\n") # dataframe correlation

    # Divide data on train and test without shuffle
    train_X = numpy.array(df.values[:divide,0:-1])
    train_Y_p = numpy.array(df.values[:divide,-1])
    test_X = numpy.array(df.values[divide:,0:-1])
    test_Y_p = numpy.array(df.values[divide:,-1])
    train_Y = train_Y_p.reshape(-1,1)
    test_Y = test_Y_p.reshape(-1,1)
    print(train_X.shape,"\n",train_Y.shape)

    return train_X, test_X, train_Y, test_Y
