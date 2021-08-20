import json  
import pandas as pd 
import numpy as np
import pickle
import tensorflow as tf 
import feature_info

INSTANCE_LIST = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']
MODEL_DICT = {}

# Load Model
for anchor_instance in INSTANCE_LIST:
	pred_instance_list = [x for x in INSTANCE_LIST if x != anchor_instance]
	for pred_instance in pred_instance_list:
		anchor_name = anchor_instance[:2]
		pred_name = pred_instance[:2]
        model_linear = pickle.load(open(f'/var/task/PROFET_WEB_TEST_YS/Model/anchor_{anchor_name}_{pred_name}_model_simple.bin', 'rb')) 
        model_rfr = pickle.load(open(f'/var/task/PROFET_WEB_TEST_YS/Model/anchor_{anchor_name}_{pred_name}_model_rfr.bin', 'rb'))
        model_dnn = tf.keras.models.load_model(f"/var/task/PROFET_WEB_TEST_YS/Model/anchor_{anchor_name}_{pred_name}_model_dnn")
		MODEL_DICT[f'{anchor_name}-{pred_name}-linear'] = model_linear
		MODEL_DICT[f'{anchor_name}-{pred_name}-rfr'] = model_rfr
		MODEL_DICT[f'{anchor_name}-{pred_name}-dnn'] = model_dnn

def feature_clustering(test_x):
	missing_columns_list = [x for x in feature_info.FEATURE_COLUMNS if x not in test_x.columns]
	for i in missing_columns_list:
		test_x[i] = 0
	for key, value in feature_info.CLUSTER_FEATURES.items():
		value = ["Device_" + x for x in value]
		test_x["&".join(value)] = 0
		for feature in value:
			test_x["&".join(value)] += test_x[feature]
		test_x.drop(value, axis=1, inplace=True)
	return test_x


def median_ensemble(test_x, anchor_latency, anchor_instance, pred_instance):
    anchor_name = anchor_instance[:2]
    pred_name = pred_instance[:2]
		
    # Predict
    rfr_pred = (MODEL_DICT[f'{anchor_name}-{pred_name}-rfr']).predict(test_x).reshape(-1, 1)
    dnn_pred = (MODEL_DICT[f'{anchor_name}-{pred_name}-dnn']).predict(test_x).reshape(-1, 1)
    median_pred = np.median(np.stack([dnn_pred, rfr_pred]), axis=0)

    if anchor_latency != 0:
        linear_pred = (MODEL_DICT[f'{anchor_name}-{pred_name}-linear']).predict(np.array(anchor_latency).reshape(-1, 1)).reshape(-1, 1)	
        median_pred = np.median(np.stack([dnn_pred, rfr_pred, linear_pred]), axis=0)

    return median_pred[0][0]
    
def lambda_handler(event, context):
    # Load Data & Parsing
    body = event['body-json']
    printbody=body

    json_feature = json.loads(body[body.find('['): body.rfind(']')+1]) 
    body = body.replace('\n',"")
    body = body.replace('\r',"")
    body = body.replace('-',"")
 
    # ANCsHOR_INSTANCE, BATCH_LATENCY, PRED_INSTANCES  Declaration
    try:
        BATCH_LATENCY = int(body[body.find('anchor_latency')+15: body.rfind('WebKitFormBoundary')])    
    except:
        BATCH_LATENCY = 0
    
    body = (body[: body.rfind('WebKitFormBoundary')])
    ANCHOR_INSTANCE = (body[body.find('anchor_instance')+16: body.rfind('WebKitFormBoundary')])
    PRED_INSTANCES = [x for x in INSTANCE_LIST if x != ANCHOR_INSTANCE] 
        
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
        'event' : printbody
    }