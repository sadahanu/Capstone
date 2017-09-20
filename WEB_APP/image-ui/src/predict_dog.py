from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import os.path
from tensorflow.python.platform import gfile
from src.retrain import add_jpeg_decoding
import glob
import pandas as pd
import boto3

def create_transfermodel_info(model_file_name):
    return {'bottleneck_tensor_name':"pool_3/_reshape:0",
    'bottleneck_tensor_size':2048,
    'input_width':299,
    'input_height':299,
    'input_depth':3,
    'resized_input_tensor_name':'Mul:0',
    'model_file_name':model_file_name,
    'input_mean':128,
    'input_std':128,
    'tmodel_weights':'final_training_ops/weights/final_weights:0',
    'tmodel_bias':'final_training_ops/biases/final_biases:0'
           }

def create_transfermodel_graph(tfmodel_info, model_directory):
    with tf.Graph().as_default() as graph:
        tfmodel_path = os.path.join(model_directory, tfmodel_info['model_file_name'])
        with gfile.FastGFile(tfmodel_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor,final_weights_tensor, final_biases_tensor = (tf.import_graph_def(
              graph_def,
              name='',
              return_elements=[
                  tfmodel_info['bottleneck_tensor_name'],
                  tfmodel_info['resized_input_tensor_name'],
                  tfmodel_info['tmodel_weights'],
                  tfmodel_info['tmodel_bias'],
              ]))
    return graph, bottleneck_tensor, resized_input_tensor,final_weights_tensor, final_biases_tensor

def get_predict_image_bottleneck(image_path, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        #bottleneck_value = run_bottleneck_on_image(
        #    sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        #    resized_input_tensor, bottleneck_tensor)
          # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(decoded_image_tensor,
                                  {jpeg_data_tensor: image_data})
          # Then run it through the recognition network.
        bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
    return bottleneck_values

def get_one_prediction(transfer_model_info, test_image_path, model_path):
    graph, bottleneck_tensor, resized_image_tensor, final_weights_tensor, \
    final_biases_tensor = (create_transfermodel_graph(transfer_model_info, model_path))
    with tf.Session(graph=graph) as sess:
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        transfer_model_info['input_width'], transfer_model_info['input_height'],
        transfer_model_info['input_depth'], transfer_model_info['input_mean'],
        transfer_model_info['input_std'])
        bo_value = get_predict_image_bottleneck(test_image_path, sess, jpeg_data_tensor,
                                            decoded_image_tensor, resized_image_tensor, bottleneck_tensor)
        bo_value = bo_value.reshape((1, bo_value.shape[0]))
        logits = tf.matmul(bo_value, final_weights_tensor) + final_biases_tensor
        pred_prob = tf.nn.softmax(logits)
        prob_res = sess.run(pred_prob)
    return prob_res

def getLabelList(label_file):
    labels = []
    with open(label_file, 'r') as f:
        labels = f.read().splitlines()
    return np.array(labels)

def getOnePredLabel(probs, labels):
    pred_labels = []
    for prob in probs:
        pred_labels.append(labels[np.argmax(prob)])
    return pred_labels

def getTop3PredLabel(probs, labels):
    pred_labels = []
    for prob in probs:
        indexes = np.argsort(-prob)[:3]
        pred_labels.append(zip(labels[indexes], prob[indexes]))
    return pred_labels

def get_dir_prediction(transfer_model_info, test_image_path, model_path):
    graph, bottleneck_tensor, resized_image_tensor, final_weights_tensor, \
    final_biases_tensor = (create_transfermodel_graph(transfer_model_info, model_path))
    with tf.Session(graph=graph) as sess:
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        transfer_model_info['input_width'], transfer_model_info['input_height'],
        transfer_model_info['input_depth'], transfer_model_info['input_mean'],
        transfer_model_info['input_std'])
        images_bottlenecks = []
        names = []
        count = 0
        for jpgpath in glob.iglob(test_image_path+"*.jpg"):
            if count%50==1:
                print("processing until "+jpgpath)
            count+=1
            bo_value = get_predict_image_bottleneck(jpgpath, sess, jpeg_data_tensor,
                                            decoded_image_tensor, resized_image_tensor, bottleneck_tensor)
                #bo_value = bo_value.reshape((1, bo_value.shape[0]))
            images_bottlenecks.append(bo_value)
            names.append(jpgpath.split('/')[-1])
        logits = tf.matmul(tf.stack(images_bottlenecks), final_weights_tensor) + \
        final_biases_tensor
        pred_prob = tf.nn.softmax(logits)
        prob_res = sess.run(pred_prob)
    return names,prob_res

def get_one_prediction_for_web(model_file_name, label_file, test_file):
    LABELS = getLabelList(label_file)
    tf_model_info = create_transfermodel_info(model_file_name)
    probs = get_one_prediction(tf_model_info, test_file, '')
    result = {}
    result['one_prediction'] = getOnePredLabel(probs, LABELS)
    result['three_predictions'] = getTop3PredLabel(probs, LABELS)
    return result

def get_recommendation_for_web(model_file_name, label_file, test_file, k=3):
    # load necessary data frame :
    df_scored  = pd.read_csv("s3://dogfaces/reviews/scored_breed_toy.csv")
    toy_df = pd.read_csv("s3://dogfaces/reviews/toys.csv")
    LABELS = getLabelList(label_file)
    # get model prediction
    tf_model_info = create_transfermodel_info(model_file_name)
    probs = get_one_prediction(tf_model_info, test_file, '')
    result = {}
    result['one_prediction'] = getOnePredLabel(probs, LABELS)
    result['three_predictions'] = getTop3PredLabel(probs, LABELS)
    breeds = getLabelList(label_file)
    # calculating recommendations
    D = df_scored.shape[1] - 1
    prob_v = probs[0].reshape((D,1))
    score_mat = df_scored[breeds].values
    fscore_mat = score_mat.dot(prob_v)
    top_ind = np.argsort(-fscore_mat[:,0])[:k]
    top_toy = df_scored['toy_id'].values[top_ind]
    likely_ratings = pd.DataFrame({"likelyrating":np.round(fscore_mat[:,0][top_ind],2)}, index=None)
    toy_info = toy_df[toy_df['toy_id'].isin(top_toy)][['toy_name','price', 'toy_link', 'picture_link']].copy()
    toy_recom = pd.concat([toy_info.reset_index(), likely_ratings], axis=1)
    return result, toy_recom

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file_name',
    type=str,
    default="output_graph.pb",
    help="name of trained model")

    parser.add_argument('--model_dir',
    type=str,
    default='',
    help='Path to Folders of model')

    parser.add_argument('--label_file',
    type=str,
    default='',
    help='Path to label of model')

    parser.add_argument('--test_file',
    type=str,
    default='',
    help="Path to Test Images")

    parser.add_argument('--predict_multiple',
    type=str,
    default='N',
    help="whether predict multiple test images Y/N")

    PARAS, _ = parser.parse_known_args()

    LABELS = getLabelList(PARAS.label_file)
    tf_model_info = create_transfermodel_info(PARAS.model_file_name)
    if PARAS.predict_multiple == 'N':
        probs = get_one_prediction(tf_model_info, PARAS.test_file, PARAS.model_dir)
        print ("+++++++++++++++++++++++++++")
        print("top 1 prediction is:")
        print(getOnePredLabel(probs, LABELS))
        print ("top3 prediction is:")
        print(getTop3PredLabel(probs, LABELS))
    else:
        names, probs = get_dir_prediction(tf_model_info, PARAS.test_file, PARAS.model_dir)
        df = pd.DataFrame({'pic_names':names, 'probability':list(probs)})
        df_data = df.to_csv(index=False, encoding='utf-8')
        s3_res = boto3.resource('s3')
        s3_res.Bucket('dogfaces').put_object(Key='reviews/labeled_pictures.csv', Body=df_data)
