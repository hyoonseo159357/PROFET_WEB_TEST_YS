import sys 
sys.path.append("/mnt/efs/packages") 
import json  
import pandas as pd 
import numpy as np
import pickle
import tensorflow as tf 
import feature_info

INSTANCE_LIST = ['g3s_xlarge', 'g4dn_xlarge', 'p2_xlarge', 'p3_2xlarge']

def median_ensemble(test_x, anchor_latency, anchor_instance, pred_instance):
    anchor_name = anchor_instance.split("_")[0]
    pred_name = pred_instance.split("_")[0]
		
    # Load model & Predict
    model_rfr = pickle.load(open(f'/mnt/efs/packages/{anchor_name}_{pred_name}_model_rfr.bin', 'rb')) 
    rfr_pred = model_rfr.predict(test_x).reshape(-1, 1)
    model_dnn = tf.keras.models.load_model(f"/mnt/efs/packages/{anchor_name}_{pred_name}_model_dnn.bin")
    dnn_pred = model_dnn.predict(test_x).reshape(-1, 1)
    median_pred = np.median(np.stack([dnn_pred, rfr_pred]), axis=0)

    if anchor_latency != 0:
        model_linear = pickle.load(open(f'/mnt/efs/packages/{anchor_name}_{pred_name}_model_simple.bin', 'rb')) 
        linear_pred = model_linear.predict(np.array(anchor_latency).reshape(-1, 1)).reshape(-1, 1)	
        median_pred = np.median(np.stack([dnn_pred, rfr_pred, linear_pred]), axis=0)

    return median_pred[0][0]
    
def lambda_handler(event, context):
    # Load Data & Parsing
    body = event['body-json']
    json_feature = json.loads(body[body.find('['): body.rfind(']')+1]) 
    body = body.replace('\n',"")
    body = body.replace('\r',"")
    body = body.replace('-',"")
 
    # ANCHOR_INSTANCE, BATCH_LATENCY, PRED_INSTANCES  Declaration
    try:
        BATCH_LATENCY = int(body[body.find('anchor_latency')+15: body.rfind('WebKitFormBoundary')])    
    except:
        BATCH_LATENCY = 0
    
    body = (body[: body.rfind('WebKitFormBoundary')])
    ANCHOR_INSTANCE = (body[body.find('anchor_instance')+16: body.rfind('WebKitFormBoundary')])
    PRED_INSTANCES = [x for x in INSTANCE_LIST if x != ANCHOR_INSTANCE] 
    
    # Append Miss Columns
    test_x = pd.DataFrame(json_feature)
    missing_columns_list = [x for x in feature_info.FEATURE_COLUMNS if x not in test_x.columns]
    for i in missing_columns_list:
        test_x[i] = 0 
    
    # Feature Aggregation 
    for key, value in feature_info.CLUSTER_FEATURES.items():
        value = ["Device_" + x for x in value]
        test_x["&".join(value)] = 0
        for feature in value:
            test_x["&".join(value)] += test_x[feature]
        test_x.drop(value, axis=1, inplace=True)
        
    # Predict
    for pred_instance in PRED_INSTANCES: 
        globals()['result_{}'.format(pred_instance)] = median_ensemble(test_x, BATCH_LATENCY, ANCHOR_INSTANCE, pred_instance)	 
	
	# Output
    result_latency = []
    result_instance = []

    for pred_instance in PRED_INSTANCES:
        result_latency.append(globals()['result_{}'.format(pred_instance)])
        result_instance.append(pred_instance)
             
    return {
        'statusCode': 200,
        'body': json.dumps(f"{result_latency[0]}&{result_latency[1]}&{result_latency[2]}&{result_instance[0]}&{result_instance[1]}&{result_instance[2]}&")
    }