FEATURE_COLUMNS = ['Device_SoftmaxCrossEntropyWithLogits', 'Host_IDLE',
                  'Device_Conv2DBackpropFilter', 'Device_RealDiv', 'Host__Send',
                  'Host_FlushSummaryWriter', 'Device_MaxPool',
                  'Device_FusedBatchNormGradV3', 'Device_AvgPoolGrad', 'Device_AddN',
                  'Device_Slice', 'Device_ReluGrad', 'Host_IteratorGetNext',
                  'Host_Identity', 'Device_AvgPool', 'Device_Relu', 'Host_GatherV2',
                  'Device_Conv2D', 'Host__HostSend', 'Device__Recv', 'Host_Dataset',
                  'Device_Transpose', 'Device_Mean', 'Device_ConcatV2',
                  'Device_MaxPoolGrad', 'Device_FusedBatchNormV3', 'Device_IDLE',
                  'Device_Conv2DBackpropInput', 'Host_LogicalAnd', 'Host_WriteSummary',
                  'Device_DepthwiseConv2dNative', 'Device_ResourceApplyGradientDescent',
                  'Device_AssignSubVariableOp', 'Device_Relu6Grad', 'Device_BiasAddGrad',
                  'Device_AddV2', 'Device_MatMul', 'Device_RsqrtGrad', 'Device_BiasAdd',
                  'Device_Pad', 'Device_Equal', 'Device_Sum', 'Device_Neg',
                  'Device_RandomUniform', 'Device_Sub',
                  'Device_DepthwiseConv2dNativeBackpropInput',
                  'Device_AssignAddVariableOp', 'Device_BroadcastTo',
                  'Device_GreaterEqual', 'Device_LogicalAnd', 'Device_Cast',
                  'Device_Softmax', 'Device_Relu6', 'Device_Mul', 'Device__HostRecv',
                  'Device_DynamicStitch', 'Device_DepthwiseConv2dNativeBackpropFilter',
                  'Device__FusedConv2D', 'Device_ArgMax', 'Device_DivNoNan',
                  'Device_Rsqrt', 'Device__Send', 'Device_SquaredDifference', 'Device_Tile', 'Device_Square']
CLUSTER_FEATURES={0: ['DepthwiseConv2dNativeBackpropInput', 'DepthwiseConv2dNativeBackpropFilter'], 
                    1: ['DepthwiseConv2dNative'], 
                    2: ['Conv2DBackpropInput', 'Conv2DBackpropFilter'], 
                    3: ['AssignSubVariableOp', 'AssignAddVariableOp'], 
                    4: ['FusedBatchNormV3', 'FusedBatchNormGradV3'],
                    5: ['BroadcastTo'], 
                    6: ['LogicalAnd'], 
                    7: ['BiasAddGrad', 'BiasAdd'], 
                    8: ['MaxPoolGrad', 'AvgPoolGrad'], 
                    9: ['Relu6Grad', 'RsqrtGrad', 'ReluGrad'], 
                    10: ['Conv2D', 'ConcatV2'], 
                    11: ['MatMul', 'MaxPool', 'AvgPool'], 
                    12: ['Softmax', 'ArgMax'], 
                    13: ['Relu', 'AddV2', 'IDLE', 'Pad', 'Mean', 'Equal', 'Sum', 'Neg', 'Sub', 'Slice', 'RealDiv', 'Cast', 'Relu6', 'Mul', '_Recv', 'Rsqrt', 'AddN', '_Send', 'Tile', 'Square'], 
                    14: ['DivNoNan'], 
                    15: ['_HostRecv'], 
                    16: ['Transpose'], 
                    17: ['GreaterEqual'], 
                    18: ['_FusedConv2D'], 
                    19: ['RandomUniform'], 
                    20: ['DynamicStitch'], 
                    21: ['SquaredDifference'], 
                    22: ['ResourceApplyGradientDescent'], 
                    23: ['SoftmaxCrossEntropyWithLogits']}