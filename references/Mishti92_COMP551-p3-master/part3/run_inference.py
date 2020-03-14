import numpy as np
import tensorflow as tf
import os.path
import os
import csv
import time
import cPickle
path = 'test/'
modelFullPath = 'model/retrained_graph.pb'
labelsFullPath = 'model/retrained_labels.txt'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None
    answers = []
    # Creates graph from saved GraphDef.
    create_graph()
    
    for i in range(6600):
#        print "inside loop", i
        imagePath=path+ str(i)+".jpeg"
#           print filename
#        if not os.path.isfile(filename):
#            print filename + " does not exist"
        if not tf.gfile.Exists(imagePath):
            tf.logging.fatal('File does not exist %s', imagePath)
#            return answer
#        print "saving image data"
        image_data = tf.gfile.FastGFile(imagePath, 'rb').read()


        with tf.Session() as sess:
#            print "Inside tf session"

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-1:][::-1]  # Getting top 5 predictions
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]
    #        for node_id in top_k:
    #            human_string = labels[node_id]
    #            score = predictions[node_id]
    #            print('%s (score = %.5f)' % (human_string, score))
    #            print

            answer = labels[top_k[0]]
            print i, ",", answer[5:]
            answers.append(answer[5:])
    return answers


if __name__ == '__main__':
#    run_inference_on_image()
#    for filename in os.listdir(path):
#        if filename.endswith(".jpeg"):
#            imagePath = os.path.join(path, filename)
#            print imagePath, filename

    test_predict_CNN_filename = "output/CNN_retrained"
    answers = []
#    
#            
    answers = run_inference_on_image()
    cPickle.dump(answers, open('answers.p', 'wb')) 
#        print i, answer
#        answers.append(answer)

    with open(test_predict_CNN_filename + time.strftime("%d_%m_%Y_%H_%M_%S") + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'class'])

        for i in range(len(answers)):
            writer.writerow([i, answers[i]])
