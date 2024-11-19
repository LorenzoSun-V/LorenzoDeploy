'''
第一步 添加节点前的操作

进行切片拼接操作

'''

import argparse
import onnx_graphsurgeon as gs
import numpy as np
import onnx


def add_postprocess(input_name, numClasses, output_name):
    graph = gs.import_onnx(onnx.load(input_name))
    assert "concat" in graph.nodes[-1].name.lower(), "last node of onnx is not concat!"

    # input
    origin_output = graph.nodes[-1]
    batch_size, num_array, _ = origin_output.outputs[0].shape # yolo shape: [batch_size, num_arry, 5+cls]
    
    starts_wh = gs.Constant("starts_wh",values=np.array([0,0,0],dtype=np.int64))
    ends_wh = gs.Constant("ends_wh",values=np.array([batch_size,num_array,4],dtype=np.int64))
    # axes = gs.Constant("axes",values=np.array([2],dtype=np.int64))
    # steps = gs.Constant("steps",values=np.array([4,1,80],dtype=np.int64))
    # split = gs.Constant("split",values=np.array([4,1,80],dtype=np.int64))

    starts_object = gs.Constant("starts_object",values=np.array([0,0,4],dtype=np.int64))
    ends_object = gs.Constant("ends_object",values=np.array([batch_size,num_array,5],dtype=np.int64))

    starts_conf = gs.Constant("starts_conf",values=np.array([0,0,5],dtype=np.int64))
    ends_conf = gs.Constant("ends_conf",values=np.array([batch_size,num_array, numClasses+5],dtype=np.int64))

    # output
    box_xywh_0 = gs.Variable(name="box_xywh_0",shape=(batch_size,num_array,4),dtype=np.float32)
    object_prob_0 = gs.Variable(name="object_prob_0",shape=(batch_size,num_array,1),dtype=np.float32)
    label_conf_0 = gs.Variable(name='label_conf_0',shape=(batch_size,num_array, numClasses),dtype=np.float32)

    # trt不支持
    # split_node = gs.Node(op="Split",inputs=[origin_output.outputs[0],split],outputs= [ box_xywh_0,object_prob_0,label_conf_0] )
    # slice
    box_xywh_node = gs.Node(op="Slice",inputs=[origin_output.outputs[0],starts_wh,ends_wh],outputs= [ box_xywh_0])
    box_prob_node = gs.Node(op="Slice",inputs=[origin_output.outputs[0],starts_object,ends_object],outputs= [ object_prob_0])
    box_conf_node = gs.Node(op="Slice",inputs=[origin_output.outputs[0],starts_conf,ends_conf],outputs= [ label_conf_0])


    # identity
    box_xywh = gs.Variable(name="box_xywh",shape=(batch_size,num_array,4),dtype=np.float32)
    object_prob = gs.Variable(name="object_prob",shape=(batch_size,num_array,1),dtype=np.float32)
    label_conf = gs.Variable(name='label_conf',shape=(batch_size,num_array, numClasses),dtype=np.float32)

    identity_node_wh = gs.Node(op="Identity",inputs=[box_xywh_0],outputs= [ box_xywh] )
    identity_node_prob = gs.Node(op="Identity",inputs=[object_prob_0],outputs= [object_prob] )
    identity_node_conf = gs.Node(op="Identity",inputs=[label_conf_0],outputs= [ label_conf] )


    print(identity_node_wh)
    print(identity_node_conf)
    # graph.nodes.extend([split_node])
    # graph.outputs = [ box_xywh,object_prob,label_conf ]
    # graph.cleanup().toposort()
    # onnx.save(gs.export_onnx(graph),"test0.onnx")
    # #-----------------------重新加载模型-------------
    # graph = gs.import_onnx(onnx.load("./test0.onnx"))
    # # 添加计算类别概率的结点
    # # input
    # origin_output = [node for node in graph.nodes ][-1]
    # print(origin_output.outputs)

    # 添加xywh->x1y1x2y2的结点

    # input
    starts_1 = gs.Constant("starts_x",values=np.array([0,0,0],dtype=np.int64))
    ends_1 = gs.Constant("ends_x",values=np.array([batch_size,num_array,1],dtype=np.int64))
    # axes_1 = gs.Constant("axes",values=np.array([2],dtype=np.int64))
    # steps_1 = gs.Constant("steps",values=np.array([1],dtype=np.int64))

    starts_2 = gs.Constant("starts_y",values=np.array([0,0,1],dtype=np.int64))
    ends_2 = gs.Constant("ends_y",values=np.array([batch_size,num_array,2],dtype=np.int64))

    starts_3 = gs.Constant("starts_w",values=np.array([0,0,2],dtype=np.int64))
    ends_3 = gs.Constant("ends_w",values=np.array([batch_size,num_array,3],dtype=np.int64))

    starts_4 = gs.Constant("starts_h",values=np.array([0,0,3],dtype=np.int64))
    ends_4 = gs.Constant("ends_h",values=np.array([batch_size,num_array,4],dtype=np.int64))

    # output
    x = gs.Variable(name="x_center",shape=(batch_size,num_array,1),dtype=np.float32)
    y = gs.Variable(name="y_center",shape=(batch_size,num_array,1),dtype=np.float32)
    w = gs.Variable(name="w",shape=(batch_size,num_array,1),dtype=np.float32)
    h = gs.Variable(name="h",shape=(batch_size,num_array,1),dtype=np.float32)

    # xywh_split_node = gs.Node(op="Split",inputs=[box_xywh],outputs= [x,y,w,h] )
    x_node = gs.Node(op="Slice",inputs=[box_xywh,starts_1,ends_1],outputs=[x])
    y_node = gs.Node(op="Slice",inputs=[box_xywh,starts_2,ends_2],outputs=[y])
    w_node = gs.Node(op="Slice",inputs=[box_xywh,starts_3,ends_3],outputs=[w])
    h_node = gs.Node(op="Slice",inputs=[box_xywh,starts_4,ends_4],outputs=[h])

    # 变换1
    # input
    div_val = gs.Constant("div_val",values=np.array([2],dtype=np.float32))
    div_val_ = gs.Constant("div_val_",values=np.array([-2],dtype=np.float32))
    # output
    w_ = gs.Variable(name="w_half_",shape=(batch_size,num_array,1),dtype=np.float32)
    wplus = gs.Variable(name="w_half_plus",shape=(batch_size,num_array,1),dtype=np.float32)
    h_ = gs.Variable(name="h_half_",shape=(batch_size,num_array,1),dtype=np.float32)
    hplus = gs.Variable(name="h_half_plus",shape=(batch_size,num_array,1),dtype=np.float32)

    w_node_ =  gs.Node(op="Div",inputs=[w,div_val_],outputs= [w_] )
    w_node_plus =  gs.Node(op="Div",inputs=[w,div_val],outputs= [wplus] )
    h_node_ =  gs.Node(op="Div",inputs=[h,div_val_],outputs= [h_] )
    h_node_plus =  gs.Node(op="Div",inputs=[h,div_val],outputs= [hplus] )

    #变换2
    # output
    x1 = gs.Variable(name="x1",shape=(batch_size,num_array,1),dtype=np.float32)
    y1 = gs.Variable(name="y1",shape=(batch_size,num_array,1),dtype=np.float32)
    x2 = gs.Variable(name="x2",shape=(batch_size,num_array,1),dtype=np.float32)
    y2 = gs.Variable(name="y2",shape=(batch_size,num_array,1),dtype=np.float32)


    x1_node =  gs.Node(op="Add",inputs=[x,w_],outputs= [x1] )
    x2_node =  gs.Node(op="Add",inputs=[x,wplus],outputs= [x2] )
    y1_node =  gs.Node(op="Add",inputs=[y,h_],outputs= [y1] )
    y2_node=  gs.Node(op="Add",inputs=[y,hplus],outputs= [y2] )

    # concat
    # output 

    boxes_0 = gs.Variable(name="boxes_0",shape=(batch_size,num_array,4),dtype=np.float32)

    # print(help(gs.Node))

    boxes_node_0 = gs.Node(op="Concat",inputs=[x1,y1,x2,y2],outputs= [boxes_0] ,attrs={"axis":2})
    # print(boxes_node_0)

    # # Unsqueeze  tensorrt不支持
    # axis_squeeze = gs.Constant("axes",values=np.array([2],dtype=np.int64))
    shapes = gs.Constant("shape",values=np.array([batch_size,num_array,1,4],dtype=np.int64))

    # output
    boxes = gs.Variable(name="boxes",shape=(batch_size,num_array,1,4),dtype=np.float32)

    # boxes_node = gs.Node(op="Unsqueeze",inputs=[boxes_0,axis_squeeze],outputs= [boxes])
    # print(boxes_node)
    boxes_node = gs.Node(op="Reshape",inputs=[boxes_0,shapes],outputs= [boxes])

    #----处理prob
    scores = gs.Variable(name="scores",shape=(batch_size,num_array, numClasses),dtype=np.float32)

    # Mul是矩阵中逐点相乘
    scores_node = gs.Node(op="Mul",inputs=[label_conf,object_prob],outputs=[scores])

    graph.nodes.extend([box_xywh_node,box_prob_node,box_conf_node,identity_node_wh,identity_node_prob,identity_node_conf,
        x_node,y_node,w_node,h_node,
        w_node_,w_node_plus,h_node_,h_node_plus,x1_node,x2_node,y1_node,y2_node,boxes_node_0,boxes_node,scores_node])
    # graph.nodes.extend([split_node,xywh_split_node,w_node_,w_node_plus,h_node_,h_node_plus,x1_node,x2_node,y1_node,y2_node,boxes_node,scores_node])

    graph.outputs = [ boxes,scores ]
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_name)


def main(args):
    add_postprocess(args.inputmodel, args.numcls, args.outputmodel)
    

if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputmodel', type=str, default='./best.onnx', help='input model path')
    parser.add_argument(
        '--outputmodel', type=str, default='./last_1.onnx', help='output model path')
    parser.add_argument('--numcls', type=int, default=1, help='class number')
         
    args, _ = parser.parse_known_args()

    main(args)

