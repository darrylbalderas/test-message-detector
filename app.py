import pandas as pd

'''
Prediction 1:
scenario_1: The word test is at least in one field
scenario_2: All fields are the same
scenario_3 = All random characters in any of the fields

Prediction 0:
scenario_4 = Messages are okay and different

'''

names = ['label', 'text']
message_data = pd.read_table("SMSSpamCollection", names=names)

scenario_1 = None
scenario_2 = None
scenario_3 = None
scenario_4 = None

