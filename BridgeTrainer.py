#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:14:22 2018

@author: johannes
"""

import sys
import getopt
import os
import math
import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import confusion_matrix_pretty_print as cp

from tensorflow import keras

# Images are getting scaled down to a certain width and height. Those are the default values.
image_input_width = 30
image_input_height = 30

# This class logs certain values during training of the neural net. More specific: The accuracy, the loss,
# and the accuracy and loss against validation data.
class AccuracyLossLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.count_epochs = 1
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.count_epochs += 1
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))

class_names = ['Balkenbrücke', 'Gitterbrücke', 'Freitragende Brücke', 'Bogenbrücke', 'Langersche Balkenbrücke', 'Hängebrücke', 'Schrägseilbrücke']

def usage():
    print ("usage text goes here")
    
def printInfoMessage(infoMessage):
    print("INFO:", infoMessage)
    
def printWarnMessage(warnMessage):
    print("WARN:", warnMessage)
    
def printErrorMessage(errorMessage):
    print("ERROR:", errorMessage)
        
# load image data from a certain directory and returns 3 tuple of datas: The train data, the test data and the validation data.
# 
def load_image_data(image_dir_path):
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    
    train_file_path = os.path.join(image_dir_path, train_file_name)
    test_file_path = os.path.join(image_dir_path, test_file_name)
    
    train_images, train_labels, train_classes_count = load_image_file(image_dir_path, train_file_path)
    test_images, test_labels, test_classes_count = load_image_file(image_dir_path, test_file_path)
    
    validation_end_index = math.ceil(0.2 * len(train_images)) #20% is the amount we take from the train data and use them
                                                              #validation data during training
    
    validation_images = train_images[:validation_end_index]
    validation_labels = train_labels[:validation_end_index]
    
    train_images = train_images[validation_end_index:]
    train_labels = train_labels[validation_end_index:]
    
    return (train_images, train_labels, train_classes_count), (test_images, test_labels, test_classes_count), (validation_images, validation_labels)
                
def load_image_file(image_dir_path, image_file_path):
    
    result_images = []
    result_labels = []
    result_classes_count = []
    
    if not os.path.exists(image_file_path):
        print('Imagefile {} does not exist.'.format(image_file_path))
        sys.exit(1)
    
    try:
        with open(image_file_path, 'r') as image_file:
            for line in image_file:
                line = line.rstrip().lstrip()
                entry = line.split(';')
                image = entry[0] + '.jpg'
                image = os.path.join(image_dir_path, image)
                label = int(entry[1])
                
                result_images.append(image)
                result_labels.append(label)
                
                if(len(result_classes_count) < label):
                    for i in range(0, label - len(result_classes_count)):
                        result_classes_count.append(0)
                
                result_classes_count[label - 1] += 1
    except IOError:
        print('Error during reading file {}'.format(image_file_path))
        sys.exit(1)
        
    #randomize the order of the data befure returning it.
    random_seed = np.random.get_state()
    np.random.shuffle(result_images)
    np.random.set_state(random_seed)
    np.random.shuffle(result_labels)
        
    return result_images, result_labels, result_classes_count

# the mapping function decodes the picture to data for the neural network
# and map a label to a specific image
def tfMappingFunction(filename, label):
    global image_input_width
    global image_input_height
    
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3);
    image_resized = tf.image.resize_images(image_decoded, [image_input_width, image_input_height]);
    image = tf.cast(image_resized, tf.float32);
    image = image / 255.0
    return image, label

def create_data_generator(imageSet, labelSet, epochs, batch_size):
    sessImages = []
    sessLabels = []
    
    tfImages = tf.constant(imageSet)
    tfLabels = tf.constant(labelSet)
    
    dataset = tf.data.Dataset.from_tensor_slices((tfImages, tfLabels));
    dataset = dataset.map(tfMappingFunction);
    dataset = dataset.batch(batch_size);
    
    batch_count = math.ceil(len(imageSet) / batch_size)
    for i in range(0, epochs):
        sess = tf.Session()
        iterator = dataset.make_one_shot_iterator()
        sessImages, sessLabels = iterator.get_next()
        for j in range(0, batch_count):
            batch = sess.run((sessImages, sessLabels))
            yield batch[0], batch[1]-1
    
def create_validation_generator(image_set, label_set, epochs, batch_size):
    sessImages = []
    sessLabels = []
    
    tfImages = tf.constant(image_set)
    tfLabels = tf.constant(label_set)
    
    dataset = tf.data.Dataset.from_tensor_slices((tfImages, tfLabels));
    dataset = dataset.map(tfMappingFunction);
    dataset = dataset.batch(batch_size);
    
    batch_count = math.ceil(len(image_set) / batch_size)
    #for i in range(0, epochs):
    while True:
        sess = tf.Session()
        iterator = dataset.make_one_shot_iterator()
        sessImages, sessLabels = iterator.get_next()
        for j in range(0, batch_count):
            batch = sess.run((sessImages, sessLabels))
            yield batch[0], batch[1]-1
    
    
def create_predict_generator(imageSet, labelSet):
    sessImages = []
    sessLabels = []
    
    tfImages = tf.constant(imageSet)
    tfLabels = tf.constant(labelSet)
    
    dataset = tf.data.Dataset.from_tensor_slices((tfImages, tfLabels));
    dataset = dataset.map(tfMappingFunction);
    dataset = dataset.batch(1);
    
    index = 0
    
    sess = tf.Session()
    while True:
        if(index % len(imageSet) == 0):
            iterator = dataset.make_one_shot_iterator()
            sessImages, sessLabels = iterator.get_next()
            index = 0
        batch = sess.run((sessImages, sessLabels))
        yield batch[0], batch[1]-1, imageSet[index]
        index += 1
    
def compile_model(architecture):
    model = None
    
    model = keras.Sequential()
    
    if architecture == 'AlexNet':
        model.add(keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4),
                                      activation=keras.activations.relu,
                                      input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                                       activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1),
                                       activation=keras.activations.relu))
        model.add(keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1),
                                       activation=keras.activations.relu))
        model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                                       activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(4096, activation=keras.activations.relu))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(4096, activation=keras.activations.relu))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(7, activation=keras.activations.softmax))
        
    elif architecture == 'cnn_extended':
        model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                     activation=keras.activations.relu, 
                     input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(32, (5, 5), activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1000, activation=keras.activations.relu))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(300, activation=keras.activations.relu))
        model.add(keras.layers.Dense(2, activation=keras.activations.softmax))
    elif architecture == 'cnn':
        model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation=keras.activations.relu, 
                     input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(64, (5, 5), activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1000, activation=keras.activations.relu))
        model.add(keras.layers.Dense(2, activation=keras.activations.softmax))
    elif architecture == 'reduced_cnn':
        model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation=keras.activations.relu, 
                     input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(64, (5, 5), activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation=keras.activations.relu))
        model.add(keras.layers.Dense(7, activation=keras.activations.softmax))
    elif architecture == 'reduced_cnn2':
        model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation=keras.activations.relu, 
                     input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation=keras.activations.relu))
        model.add(keras.layers.Dense(7, activation=keras.activations.softmax))
    elif architecture == 'dense':
        model.add(keras.layers.Flatten(input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.Dense(500, input_dim = image_input_width * image_input_height * 3, activation=keras.activations.relu))
        model.add(keras.layers.Dense(300, activation=keras.activations.relu))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(7, activation=keras.activations.softmax))
    elif architecture == 'reduced_dense':
        model.add(keras.layers.Flatten(input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.Dense(50, input_dim = image_input_width * image_input_height * 3, activation=keras.activations.relu))
        model.add(keras.layers.Dense(30, activation=keras.activations.relu))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(7, activation=keras.activations.softmax))
    elif architecture == '1l_dense':
        model.add(keras.layers.Flatten(input_shape=(image_input_width, image_input_height, 3)))
        model.add(keras.layers.Dense(7, activation=keras.activations.softmax))
    elif architecture == 'rnn':
        model.add(keras.layers.SimpleRNN(7, input_shape=(1, 1, 1)))
    else:
        print('Unknown architecture {}. Allowed are cnn, reduced_cnn, dense and rnn.'.format(architecture))
        sys.exit(1)
    
    return model
    
    
def train_model(model, train_data, validation_data, batch_size, epochs):
    train_images = train_data[0]
    train_labels = train_data[1]
    validation_images = validation_data[0]
    validation_labels = validation_data[1]
    
    datagen = create_data_generator(train_images, train_labels, epochs, batch_size)
    validation_gen = create_validation_generator(validation_images, validation_labels, epochs, batch_size)
    
    epoch_steps = math.ceil(len(train_images) / batch_size)
    val_steps = math.ceil(len(validation_images) / batch_size)
    
    logger = AccuracyLossLogger()
    early_stopper = keras.callbacks.EarlyStopping(patience=50)
    model.fit_generator(datagen, steps_per_epoch=epoch_steps, validation_data=validation_gen, validation_steps=val_steps, epochs=epochs, verbose=1, callbacks=[logger, early_stopper])
    
    datagen.close()
    validation_gen.close()
    
    return logger
        

def predict(model, test_data, batch_size, epochs, predictions_file_path, predictions_picture_path, conf_matrix_picture_path):
    test_images = test_data[0]
    test_labels = test_data[1]
    
    datagen = create_data_generator(test_images, test_labels, epochs, batch_size)
    datagen2 = create_predict_generator(test_images, test_labels)
    
    steps = math.ceil(len(test_images) / batch_size)
    predictions = model.predict_generator(datagen, steps=steps, verbose=1)
    
    count_correct = 0
    count_wrong = 0
    
    class_count = len(class_names)
    class_predictions = [[0] * class_count for x in range(0, class_count)]
    
    countTestImages = len(test_images)
    rowCount = math.ceil(countTestImages / 20)
    
    predictions_file_content = "Anzahl Datensätze: " + str(countTestImages) + "\n"
    
    predicted_label_array = []
    fig = plt.figure(figsize=(60, rowCount*3), dpi=72)
    for i in range(0, countTestImages):
        image, actual_label, filename = next(datagen2)
        filename = os.path.basename(filename)
        actual_label = actual_label[0]
        
        plt.subplot(rowCount, 20, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image[0])
        
        predicted_label = np.argmax(predictions[i])
        predicted_label_array.append(predicted_label + 1)
        color = ""
        
        class_predictions[actual_label][predicted_label] += 1
        
        if class_names[predicted_label] == class_names[actual_label]:
            color = "green"
            count_correct += 1
        else:
            color = "red"
            count_wrong += 1
        label = "{}\n{} ({})".format(filename, class_names[predicted_label], class_names[actual_label])
        predictions_file_content += label + "\n"
        plt.xlabel(label, color=color)
        
    predictions_file_content += "\n"
    for i in range(0, class_count):
        predictions_file_content += "###" + class_names[i] + "###" + "\n"
        for j in range(0, class_count):
            predictions_file_content += class_names[i] + " als " + class_names[j] + " klassifiziert: " + str(class_predictions[i][j]) + "\n"
        predictions_file_content += "\n"
    
    predictions_file_content += "Richtig: {}\n".format(count_correct)
    predictions_file_content += "Falsch: {}\n".format(count_wrong)
    
    fig.savefig(predictions_picture_path, bbox_inches="tight")
    with open(predictions_file_path, 'w') as predictions_file:
        predictions_file.write(predictions_file_content)
        
    plt.close(fig)
    
    fig = plt.figure()
    #cp.plot_confusion_matrix_from_data(test_labels, predicted_label_array, columns=['Balkenbrücke', 'Bogenbrücke'])
    cp.plot_confusion_matrix_from_data(test_labels, predicted_label_array, columns=class_names)
    fig.savefig(conf_matrix_picture_path)
    plt.close(fig)
    
    datagen.close()
    datagen2.close()
    
def load_model(modelFilePath):
    temp, extension = os.path.splitext(modelFilePath)
    loadFunction = None
    
    if(extension == '.yaml'):
        loadFunction = keras.models.model_from_yaml
    elif(extension == '.json'):
        loadFunction = keras.models.model_from_json
    else:
        printWarnMessage("Unknown file extension: {}. Assume .json.".format(extension))
        loadFunction = keras.models.model_from_json
    
    with open(modelFilePath, 'r') as model_file:
        model_file_content = model_file.read()
    
    result = loadFunction(model_file_content)
    return result

def save_model(model, save_path):
    model_json = model.to_json()
    with open(save_path, 'w') as model_file:
        model_file.write(model_json)

def classes_count_to_str(classes_count):
    result = ""
    for i in range(0, len(classes_count)):
        result += "Davon {}: {}\n".format(class_names[i], classes_count[i])
    
    return result

def write_metadata(notes_file_path, train_data, test_data):
    create_date = datetime.datetime.now()
    
    train_images = train_data[0]
    train_classes_count = train_data[2]
    
    test_images = test_data[0]
    test_classes_count = test_data[2]
    
    dataset_count = len(train_images) + len(test_images)
    classes_count = []
    for i in range(0, len(train_classes_count)):
        classes_count.append(train_classes_count[i] + test_classes_count[i])
    
    count_data_set = str(dataset_count) + "\n"
    count_data_set += classes_count_to_str(classes_count)
    count_train_set = str(len(train_images)) + "\n"
    count_train_set += classes_count_to_str(train_classes_count)
    count_test_set = str(len(test_images)) + "\n"
    count_test_set += classes_count_to_str(test_classes_count)
    
    notes_content = "Datum: {}\n\nAnzahl Dantesätze: {}\nDavon Training: {}\nDavon Test: {}\n".format(create_date, count_data_set, count_train_set, count_test_set)
    write_notes(notes_file_path, notes_content)

def write_notes(notes_file_path, content):
    with open(notes_file_path, 'a') as notes_file:
        notes_file.write(content)

def save_plots(logger, model_dir):
    
    acc_picture_path = os.path.join(model_dir, 'acc.png')
    loss_picture_path = os.path.join(model_dir, 'loss.png')
    val_acc_picture_path = os.path.join(model_dir, 'val_acc.png')
    val_loss_picture_path = os.path.join(model_dir, 'val_loss.png')
    epochs = logger.count_epochs
    
    fig = plt.figure()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.plot(range(1, epochs), logger.acc)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    fig.savefig(acc_picture_path, bbox_inches="tight")
    plt.close(fig)
    
    fig = plt.figure()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.plot(range(1, epochs), logger.loss)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    fig.savefig(loss_picture_path, bbox_inches="tight")
    plt.close(fig)
    
    fig = plt.figure()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.plot(range(1, epochs), logger.val_acc)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()
    fig.savefig(val_acc_picture_path, bbox_inches="tight")
    plt.close(fig)
    
    fig = plt.figure()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.plot(range(1, epochs), logger.val_loss)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.show()
    fig.savefig(val_loss_picture_path, bbox_inches="tight")
    plt.close(fig)

def main():
    options = 'p:d:w:h:e:b:ta:l:o:'
    longOptions = ['imagepath=', 'modelDir=', 'imageWidth=', 'imageHeight=', 'epochs=', 'batchSize=', 'retrain', 'architecture=', 'learning-rate=', 'optimizer=']
    
    global image_input_width
    global image_input_height

    model = None
    model_dir = None
    
    model_file_name = 'model.json'
    weights_file_name = 'weights.h5'
    model_plot_name = 'model.svg'
    predictions_file_name = 'predictions.txt'
    predictions_picture_name = 'predictions.png'
    conf_matrix_file_name = 'conf_matrix.png'
    notes_file_name = "notes.txt"

    retrain = False
    learning_rate = 0.01
    optimizer = 'sgd'
    architecture = 'cnn'
    image_dir_path = None
    batch_size = 8
    epochs = 20

    try:
        opts, args = getopt.getopt(sys.argv[1:], options, longOptions)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ('-p', '--imagepath'):
            image_dir_path = arg
            if not os.path.isdir(image_dir_path):
                print("No directory", image_dir_path, "exists")
                sys.exit(1)
        elif opt in ('-d', '--modelDir='):
            model_dir = arg
        elif opt in ('-w', '--imageWidth='):
            image_input_width = int(arg)
        elif opt in ('-h', '--imageHeight='):
            image_input_height = int(arg)
        elif opt in ('-e', '--epochs='):
            epochs = int(arg)
        elif opt in ('-b', '--batchSize='):
            batch_size = int(arg)
        elif opt in ('-t', '--retrain'):
            retrain = True
        elif opt in ('-l', '--learning-rate='):
            learning_rate = float(arg)
        elif opt in ('-o', '--optimizer='):
            optimizer = arg
        elif opt in ('-a', '--architecture='):
            architecture = arg
        else:
            usage()
            sys.exit(2)
            
    if model_dir == None:
        model_dir = '{}_{}x{}_e{}_b{}_lr{}_o{}_{}'
        image_dir_name = os.path.basename(image_dir_path)
        model_dir = model_dir.format(architecture, image_input_width, image_input_height, epochs, batch_size,
                                     learning_rate, optimizer, image_dir_name)
    
    if not os.path.exists(model_dir):
        printInfoMessage("Model directory {} does not exist. Create it".format(model_dir))
        os.mkdir(model_dir)
    
    if(optimizer == 'sgd'):
        optimizer = keras.optimizers.SGD(lr=learning_rate)
    elif(optimizer == 'adam'):
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    elif(optimizer == 'rmsprop'):
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    else:
        print('Unknown optimizer {}.'.format(optimizer))
        sys.exit(1)
    
    model_file_path = os.path.join(model_dir, model_file_name)
    weights_file_path = os.path.join(model_dir, weights_file_name)
    plot_file_path = os.path.join(model_dir, model_plot_name)
    predictions_file_path = os.path.join(model_dir, predictions_file_name)
    predictions_picture_path = os.path.join(model_dir, predictions_picture_name)
    notes_file_path = os.path.join(model_dir, notes_file_name)
    conf_matrix_file_path = os.path.join(model_dir, conf_matrix_file_name)
    
    train_data, test_data, validation_data = load_image_data(image_dir_path)
    write_metadata(notes_file_path, train_data, test_data)
    
    if not os.path.exists(model_file_path):
        model = compile_model(architecture)
        save_model(model, model_file_path)
        keras.utils.plot_model(model, to_file=plot_file_path, show_shapes=True)
    else:
        model = load_model(model_file_path)
        
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    
    write_notes(notes_file_path, 'Loss function: {}\n'.format('sparse categorical crossentropy'))
    write_notes(notes_file_path, 'Optimizer: {}\n'.format('SGD(lr=0.01)'))
    
    if not os.path.exists(weights_file_path) or retrain:
        
        start = datetime.datetime.now()
        log = train_model(model, train_data, validation_data, batch_size, epochs)
        end = datetime.datetime.now()
        diff = end - start
        write_notes(notes_file_path, 'Trainingsdauer: ' + str(diff) + '\n')
        write_notes(notes_file_path, 'Batchgröße: ' + str(batch_size) + '\n')
        write_notes(notes_file_path, 'Epochs: ' + str(epochs) + '\n')
        model.save_weights(weights_file_path)
        save_plots(log, model_dir)
    else:
        model.load_weights(weights_file_path)
        
    predict(model, test_data, batch_size, epochs, predictions_file_path, predictions_picture_path, conf_matrix_file_path)
    
main()