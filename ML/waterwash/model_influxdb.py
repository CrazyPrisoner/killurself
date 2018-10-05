import time
import numpy
import pandas
import tensorflow as tf

from influxdb import InfluxDBClient, DataFrameClient
from datetime import tzinfo, timedelta, datetime, date

from grpc.beta import implementations
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

host1 = '192.168.4.33'
port1 = 8086
user1 = ''
password1 = ''
db_name1 = 'Labview'

USER = 'test'
PASSWORD = '12345'
DBNAME = 'example'
HOST = '192.168.4.33'
PORT = 8086

tf.app.flags.DEFINE_string('server', 'localhost:6660',
                         'inception_inference service host:port')
FLAGS = tf.app.flags.FLAGS

def main():
    # Connect to server
    client_reader = InfluxDBClient(host=host1, port=port1, username=user1, password=password1, database=db_name1)
    print("connect to Influxdb", db_name1, host1, port1)
    client_writer = InfluxDBClient(host=HOST, port=PORT, username=USER, password=PASSWORD, database=DBNAME)
    print("connect to Influxdb", DBNAME, HOST, PORT)
    # Time
    dt1 = datetime.now()
    dt1 = dt1 - timedelta(hours=6)
    dt = dt1.isoformat()
    # Query to database
    data_indb = DataFrameClient(host=host1, \
                                       username=user1, \
                                       password=password1, \
                                       database=db_name1)

    data_reader = data_indb.query('SELECT "gas_fuel_flow_x", "ngp", "npt", "t1_temperature", "t5_average_temperature", "turb_air_inlet_filter_dp", "engine_pcd" FROM "Labview"."autogen"."unit1" WHERE time > now() - 5m')
    df = pandas.DataFrame(data_reader['unit1'])
    print(df)
    index = df.index
    empty_df = pandas.DataFrame(columns=['engine_pcd', 'ngp', 'npt', 't1_temperature', 't5_average_temperature', 'turb_air_inlet_filter_dp','gas_fuel_flow_x','prediction'])
    empty_df.engine_pcd = df['engine_pcd']
    empty_df.ngp = df['ngp']
    empty_df.npt = df['npt']
    empty_df.t1_temperature = df['t1_temperature']
    empty_df.t5_average_temperature = df['t5_average_temperature']
    empty_df.turb_air_inlet_filter_dp = df['turb_air_inlet_filter_dp']
    empty_df.gas_fuel_flow_x= df['gas_fuel_flow_x']
    strafe = numpy.array(empty_df.values[:,0:-2])
    out_pp = numpy.float32(strafe)
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deka'
    request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
    request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(out_pp))
    request.output_filter.append('outputs')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    prediction = stub.Predict(request, 5.0)  # 5 secs timeout
    floats = prediction.outputs['outputs'].float_val
    predicted_array = numpy.array(floats)
    empty_df.prediction = predicted_array
    print(predicted_array)
    #print(empty_df)
    json_body = [{
         "measurement": "test_waterwash",
         "tags": {"type": "ww"},
         "time": dt,
         "fields": {"time": empty_df.index[-1],
              "engine_pcd" : empty_df.engine_pcd[-1],
              "ngp": empty_df.ngp[-1],
              "npt": empty_df.npt[-1],
              "t1_temperature": empty_df.t1_temperature[-1],
              "t5_average_temperature": empty_df.t5_average_temperature[-1],
              "turb_air_inlet_filter_dp": empty_df.turb_air_inlet_filter_dp[-1],
              "gas_fuel_flow_x": empty_df.gas_fuel_flow_x[-1],
              "prediction": empty_df.prediction[-1]}
    }]
    client_writer.write_points(json_body, database="example")
    #print(json_body)

    client_reader.close()
    client_writer.close()

if __name__ == '__main__':
    main()
