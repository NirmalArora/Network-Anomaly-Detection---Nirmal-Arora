import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the trained models
with open('binary_classification_dt_model.pkl', 'rb') as binary_model_file:
    binary_model = pickle.load(binary_model_file)

with open('category_classification_dt_model.pkl', 'rb') as multiclass_model_file:
    multiclass_model = pickle.load(multiclass_model_file)

# Load the scaler and encoders used during model training. 
with open('bin_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('bin_label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Define the feature categories
protocoltype_values = ['tcp', 'udp', 'icmp']
service_values = ['ftp_data', 'other', 'private', 'http', 'remote_job', 'name', 'netbios_ns', 'eco_i', 'mtp',
                  'telnet', 'finger', 'domain_u', 'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp', 'netbios_dgm',
                  'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap', 'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 
                  'whois', 'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login', 'kshell', 'sql_net', 'time', 
                  'hostnames', 'exec', 'ntp_u', 'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell', 'netstat', 
                  'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i', 'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 
                  'urh_i', 'http_8001', 'aol', 'http_2784', 'tftp_u', 'harvest']
flag_values = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH']
lastflag_values = [20, 15, 19, 21, 18, 17, 16, 12, 14, 11, 2, 13, 10, 9, 8, 7, 3, 5, 1, 6, 0, 4]

# Title of the app
st.title("Network Anomaly Detection")

# Input fields for the required columns
st.write("### Enter the network data:")
columns=['duration', 'protocoltype', 'service', 'flag', 'srcbytes', 'dstbytes',
       'land', 'wrongfragment', 'urgent', 'hot', 'numfailedlogins', 'loggedin',
       'numcompromised', 'rootshell', 'suattempted', 'numroot',
       'numfilecreations', 'numshells', 'numaccessfiles',
       'ishostlogin', 'isguestlogin', 'count', 'srvcount', 'serrorrate',
       'srvserrorrate', 'rerrorrate', 'srvrerrorrate', 'samesrvrate',
       'diffsrvrate', 'srvdiffhostrate', 'dsthostcount', 'dsthostsrvcount',
       'dsthostsamesrvrate', 'dsthostdiffsrvrate', 'dsthostsamesrcportrate',
       'dsthostsrvdiffhostrate', 'dsthostserrorrate', 'dsthostsrvserrorrate',
       'dsthostrerrorrate', 'dsthostsrvrerrorrate', 'lastflag']

# columns = [
#     'duration', 'protocoltype', 'service', 'flag', 'srcbytes', 'dstbytes', 'land',
#     'wrongfragment', 'urgent', 'hot', 'numfailedlogins', 'loggedin', 'numcompromised',
#     'rootshell', 'suattempted', 'numroot', 'numfilecreations', 'numshells', 'numaccessfiles',
#     'ishostlogin', 'isguestlogin', 'count', 'srvcount', 'serrorrate', 'srvserrorrate',
#     'rerrorrate', 'srvrerrorrate', 'samesrvrate', 'diffsrvrate', 'srvdiffhostrate',
#     'dsthostcount', 'dsthostsrvcount', 'dsthostsamesrvrate', 'dsthostdiffsrvrate',
#     'dsthostsamesrcportrate', 'dsthostsrvdiffhostrate', 'dsthostserrorrate', 'dsthostsrvserrorrate',
#     'dsthostrerrorrate', 'dsthostsrvrerrorrate', 'lastflag'
# ]
# flag_categories_dict = {
#     "Success Flag": ["SF"],
#     "S Flag": ["S0", "S1", "S2", "S3"],
#     "R Flag": ["REJ"],
#     "Reset Flag": ["RSTR", "RSTO", "RSTOS0"],
#     "SH&Oth Flag": ["SH","OTH"]
# }

service_categories_dict = {
    "Remote Access and Control Services": [
        "telnet", "ssh", "login", "kshell", "klogin", "remote_job", "rje", "shell", "supdup"
    ],
    "File Transfer and Storage Services": [
        "ftp", "ftp_data", "tftp_u", "uucp", "uucp_path", "pm_dump", "printer"
    ],
    "Web and Internet Services": [
        "http", "http_443", "http_2784", "http_8001", "gopher", "whois", "Z39_50", "efs"
    ],
    "Email and Messaging Services": [
        "smtp", "imap4", "pop_2", "pop_3", "IRC", "nntp", "nnsp"
    ],
    "Networking Protocols and Name Services": [
        "domain", "domain_u", "netbios_dgm", "netbios_ns", "netbios_ssn", "ntp_u", "name", "hostnames"
    ],
    "Database and Directory Services": [
        "ldap", "sql_net"
    ],
    "Error and Diagnostic Services": [
        "echo", "discard", "netstat", "systat"
    ],
    "Miscellaneous and Legacy Services": [
        "aol", "auth", "bgp", "csnet_ns", "daytime", "exec", "finger", "time", 
        "tim_i", "urh_i", "urp_i", "vmnet", "sunrpc", "iso_tsap", "ctf", 
        "mtp", "link", "harvest", "courier", "X11", "red_i", 
        "eco_i", "ecr_i", "other", "private"
    ]
}
# service_to_category = {service: service_category for service_category, services in service_categories_dict.items() for service in services}
# flag_to_category = {flag: flag_category for flag_category, flags in flag_categories_dict.items() for flag in flags}
# def map_service_to_category(service):
#     return service_to_category.get(service, "NAN")  # Default to "NAN"
# def map_flag_to_category(flag):
#     return flag_to_category.get(flag, "NAN")  # Default to "NAN"

# Gather inputs
user_input = {}
for col in columns:
    if col == 'protocoltype':
        user_input[col] = st.selectbox(f"{col}:", protocoltype_values)
    elif col == 'service':
        user_input[col] = st.selectbox(f"{col}:", service_values)
    elif col == 'flag':
        user_input[col] = st.selectbox(f"{col}:", flag_values)
    elif col == 'lastflag':
        user_input[col] = st.selectbox(f"{col}:", lastflag_values)
    else:
        user_input[col] = st.number_input(col, value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])



# Feature Engineering (same steps as in training)
#input_df['service_category'] = input_df['service'].apply(map_service_to_category)
#input_df['flag_category'] = input_df['flag'].apply(map_flag_to_category)
##
input_df['serrors_count'] = input_df['serrorrate']*input_df['count']
input_df['rerrors_count'] = input_df['rerrorrate']*input_df['count']

input_df['samesrv_count'] = input_df['samesrvrate']*input_df['count']
input_df['diffsrv_count'] = input_df['diffsrvrate']*input_df['count']

input_df['serrors_srvcount'] = input_df['srvserrorrate']*input_df['srvcount']
input_df['rerrors_srvcount'] = input_df['srvrerrorrate']*input_df['srvcount']

input_df['srvdiffhost_srvcount'] = input_df['srvdiffhostrate']*input_df['srvcount']

input_df['dsthost_serrors_count'] = input_df['dsthostserrorrate']*input_df['dsthostcount']
input_df['dsthost_rerrors_count'] = input_df['dsthostrerrorrate']*input_df['dsthostcount']

input_df['dsthost_samesrv_count'] = input_df['dsthostsamesrvrate']*input_df['dsthostcount']
input_df['dsthost_diffsrv_count'] = input_df['dsthostdiffsrvrate']*input_df['dsthostcount']

input_df['dsthost_serrors_srvcount'] = input_df['dsthostsrvserrorrate']*input_df['dsthostsrvcount']
input_df['dsthost_rerrors_srvcount'] =input_df['dsthostsrvrerrorrate']*input_df['dsthostsrvcount']

input_df['dsthost_samesrcport_srvcount'] = input_df['dsthostsamesrcportrate']*input_df['dsthostsrvcount']
input_df['dsthost_srvdiffhost_srvcount'] = input_df['dsthostsrvdiffhostrate']*input_df['dsthostsrvcount']
##add speed features
input_df['srcbytes/sec'] = input_df.apply(
    lambda row: row['srcbytes'] / row['duration'] if row['duration'] != 0 else row['srcbytes'] / (row['duration'] + 0.001), 
    axis=1
)
input_df['dstbytes/sec'] = input_df.apply(
    lambda row: row['dstbytes'] / row['duration'] if row['duration'] != 0 else row['dstbytes'] / (row['duration'] + 0.001), 
    axis=1
)
#modify suattpented such that it is binary
input_df["suattempted"] = input_df["suattempted"].apply(lambda x: 0 if x == 0 else 1)
# Scaling numerical features
input_df[['duration', 'srcbytes', 'dstbytes',
       'land', 'wrongfragment', 'urgent', 'hot', 'numfailedlogins', 'loggedin',
       'numcompromised', 'rootshell', 'numroot',
       'numfilecreations', 'numshells', 'numaccessfiles', 'ishostlogin',
       'isguestlogin', 'count', 'srvcount', 'serrorrate', 'srvserrorrate',
       'rerrorrate', 'srvrerrorrate', 'samesrvrate', 'diffsrvrate',
       'srvdiffhostrate', 'dsthostcount', 'dsthostsrvcount',
       'dsthostsamesrvrate', 'dsthostdiffsrvrate', 'dsthostsamesrcportrate',
       'dsthostsrvdiffhostrate', 'dsthostserrorrate', 'dsthostsrvserrorrate',
       'dsthostrerrorrate', 'dsthostsrvrerrorrate', 'serrors_count', 'rerrors_count', 'samesrv_count',
       'diffsrv_count', 'serrors_srvcount', 'rerrors_srvcount',
       'srvdiffhost_srvcount', 'dsthost_serrors_count',
       'dsthost_rerrors_count', 'dsthost_samesrv_count',
       'dsthost_diffsrv_count', 'dsthost_serrors_srvcount',
       'dsthost_rerrors_srvcount', 'dsthost_samesrcport_srvcount',
       'dsthost_srvdiffhost_srvcount', 'srcbytes/sec', 'dstbytes/sec']] = scaler.transform(
    input_df[['duration', 'srcbytes', 'dstbytes',
       'land', 'wrongfragment', 'urgent', 'hot', 'numfailedlogins', 'loggedin',
       'numcompromised', 'rootshell', 'numroot',
       'numfilecreations', 'numshells', 'numaccessfiles', 'ishostlogin',
       'isguestlogin', 'count', 'srvcount', 'serrorrate', 'srvserrorrate',
       'rerrorrate', 'srvrerrorrate', 'samesrvrate', 'diffsrvrate',
       'srvdiffhostrate', 'dsthostcount', 'dsthostsrvcount',
       'dsthostsamesrvrate', 'dsthostdiffsrvrate', 'dsthostsamesrcportrate',
       'dsthostsrvdiffhostrate', 'dsthostserrorrate', 'dsthostsrvserrorrate',
       'dsthostrerrorrate', 'dsthostsrvrerrorrate', 'serrors_count', 'rerrors_count', 'samesrv_count',
       'diffsrv_count', 'serrors_srvcount', 'rerrors_srvcount',
       'srvdiffhost_srvcount', 'dsthost_serrors_count',
       'dsthost_rerrors_count', 'dsthost_samesrv_count',
       'dsthost_diffsrv_count', 'dsthost_serrors_srvcount',
       'dsthost_rerrors_srvcount', 'dsthost_samesrcport_srvcount',
       'dsthost_srvdiffhost_srvcount', 'srcbytes/sec', 'dstbytes/sec']])
# Encoding categorical features
for col in ['protocoltype', 'service', 'flag']:
    input_df[col] = label_encoders[col].transform(input_df[col])
    ### re-arrange the columns as per training code
input_df = input_df[[
    'protocoltype', 'service', 'flag', 'lastflag', 'suattempted',
    'duration', 'srcbytes', 'dstbytes', 'land', 'wrongfragment', 'urgent',
    'hot', 'numfailedlogins', 'loggedin', 'numcompromised', 'rootshell',
    'numroot', 'numfilecreations', 'numshells', 'numaccessfiles',
    'ishostlogin', 'isguestlogin', 'count', 'srvcount', 'serrorrate',
    'srvserrorrate', 'rerrorrate', 'srvrerrorrate', 'samesrvrate',
    'diffsrvrate', 'srvdiffhostrate', 'dsthostcount', 'dsthostsrvcount',
    'dsthostsamesrvrate', 'dsthostdiffsrvrate', 'dsthostsamesrcportrate',
    'dsthostsrvdiffhostrate', 'dsthostserrorrate', 'dsthostsrvserrorrate',
    'dsthostrerrorrate', 'dsthostsrvrerrorrate', 'serrors_count',
    'rerrors_count', 'samesrv_count', 'diffsrv_count', 'serrors_srvcount',
    'rerrors_srvcount', 'srvdiffhost_srvcount', 'dsthost_serrors_count',
    'dsthost_rerrors_count', 'dsthost_samesrv_count',
    'dsthost_diffsrv_count', 'dsthost_serrors_srvcount',
    'dsthost_rerrors_srvcount', 'dsthost_samesrcport_srvcount',
    'dsthost_srvdiffhost_srvcount', 'srcbytes/sec', 'dstbytes/sec'
]]

st.write("### Input Data:")
st.dataframe(input_df)
st.write(f"Number of columns in the input data: {input_df.shape[1]}")
# Predictions
if st.button("Predict"):
    # Binary Classification
    binary_prediction = binary_model.predict(input_df)
    binary_result = "Attack" if binary_prediction[0] == 1 else "Normal"
    st.write(f"**Binary Classification (Attack or Not):** {binary_result}")

    #if binary_prediction[0] == 1:
        # Multiclass Classification
    multiclass_prediction = multiclass_model.predict(input_df)
    st.write(f"**Attack Type Prediction:** {multiclass_prediction[0]}")
