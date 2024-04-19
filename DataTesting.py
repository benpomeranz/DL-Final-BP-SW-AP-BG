from openeew.data.aws import AwsDataClient

data_client = AwsDataClient('mx')

start_date_utc = '2018-02-16 23:39:00'
end_date_utc = '2018-02-16 23:42:00'

records = data_client.get_filtered_records(
  start_date_utc,
  end_date_utc
  )