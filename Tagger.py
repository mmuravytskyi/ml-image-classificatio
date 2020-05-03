from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.resnet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.layers import Input

import numpy as np
import piexif.helper
import os
import time
import concurrent.futures
import tensorflow as tf


class Tagger:
    """
    This class is designed for image classification featuring ML and pretrained CNN model ResNet.
    """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    input_image_size = (400, 400, 3)
    input_tensor = Input(shape=input_image_size)

    model = ResNet101V2(weights='imagenet', include_top=True, input_tensor=input_tensor)

    @staticmethod
    def set_meta_tag(file_path, comment):
        """
        Function sets label to an UserComment tag [EXIF metadata] of an image.

        :param file_path: Path to the file,
        :param comment: Value of a label
        """
        exif_dict = piexif.load(file_path)
        user_comment = piexif.helper.UserComment.dump(str(str(comment)))
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, file_path)

    @staticmethod
    def tag_file(file_name):
        """
        This method classifies one given image.
        :param file_name: PAth to the file, that should be classified.
        :return:
        """

        f = open('labels.txt', 'w+')

        if file_name.endswith('.jpeg') or file_name.endswith('.jpg'):

            # preprocess an image
            img = tf_image.load_img(file_name, target_size=Tagger.input_image_size[:2])
            img = tf_image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            # apply NN and make a prediction
            predictions = Tagger.model.predict(img)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)

            translated_predictions = decode_predictions(predictions, top=2)[0]
            # TODO co to za próg 0.08?
            if float(translated_predictions[0][2]) - float(translated_predictions[1][2]) <= 0.08:
                res = [translated_predictions[0][1], translated_predictions[1][1]]
                print(file_name + " few labels have been found:  " + str([i[1] for i in translated_predictions]))
                tf.keras.backend.print_tensor(Tagger.model.layers[-1].output)
                print(Tagger.model.layers[-1].output)
            else:
                res = [translated_predictions[0][1]]
                print(file_name + " one label has been found:  " + str(translated_predictions[0][1]))
                print(Tagger.model.layers[-1].output)

        f.close()
        return res

    @staticmethod
    def tag_dir(dir_path='./images', set_meta=False, file_log=False):
        """Use ML to classify your images.

        :param dir_path: Path to the directory where images for classification are located. Default: test_image_set file.
        :return: Sets 1 or multi labels, sets meta-tag with labeled class and creates output file LABELS.TXT with labels.
        """

        def thread_function(files_range):
            """Function for operations being executed inside a thread.

            :param files_range: Set on which files actions should be performed.
            :return:
            """
            if file_log: f = open('labels.txt', 'w+')
            i = 0

            time_start = time.time()
            for file_name in files_range:
                if file_name.endswith('.jpeg') or file_name.endswith('.jpg'):
                    img_path = dir_path + file_name
                    # preprocess an image
                    img = tf_image.load_img(img_path, target_size=Tagger.input_image_size[:2])
                    img = tf_image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)

                    # apply NN and make a prediction
                    predictions = Tagger.model.predict(img)
                    # decode the results into a list of tuples (class, description, probability)
                    # (one such list for each sample in the batch)
                    i += 1
                    translated_predictions = decode_predictions(predictions, top=2)[0]
                    if float(translated_predictions[0][2]) <= 0.05:
                        print(file_name + " - no label has been found, setting 'none'.")
                        if set_meta: Tagger.set_meta_tag(img_path, 'none')
                        if file_log: f.write(str(i) + '.  ' + str(file_name) + ' ' * 5 + '->  none\n')
                    else:
                        if float(translated_predictions[0][2]) - float(translated_predictions[1][2]) <= 0.08:
                            print(file_name + " ->  " + str(translated_predictions[0][1]) + ' (' + str(
                                translated_predictions[0][2]) + '), ' + str(translated_predictions[1][1]) + ' (' + str(
                                translated_predictions[1][2]) + ').')

                            if set_meta: Tagger.set_meta_tag(img_path,
                                                             translated_predictions[0][1] + ', ' +
                                                             translated_predictions[1][1])
                            if file_log: f.write(
                                str(i) + '.  ' + str(file_name) + ' ' * 5 + '->  ' + str(
                                    translated_predictions[0][1]) + ', '
                                + str(translated_predictions[1][1]) + '\n')
                        else:
                            print(file_name + " ->  " + str(translated_predictions[0][1]) + ' (' + str(
                                translated_predictions[0][2]) + ').')
                            if set_meta: Tagger.set_meta_tag(img_path, translated_predictions[0][1])
                            if file_log: f.write(
                                str(i) + '.  ' + str(file_name) + ' ' * 5 + '->  ' + str(
                                    translated_predictions[0][1]) + '\n')
            if file_log: f.close()
            print("Time: " + str(time.time() - time_start))

        if dir_path[-1] != '/':
            print('The directory path is incorrect, will be corrected!')
            dir_path = dir_path + '/'

        files_num = len(
            [name for name in os.listdir(dir_path) if
             os.path.isfile(os.path.join(dir_path, name)) and (name.endswith('.jpeg') or name.endswith('.jpg'))])
        print(str(files_num) + ' images has been detected')

        if 20 < files_num <= 50:
            """ 
            2 threads
            """
            print('Starting 2 threads.')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                executor.map(thread_function,
                             [os.listdir(dir_path)[0: files_num // 2], os.listdir(dir_path)[files_num // 2: files_num]])
        else:
            if 50 < files_num <= 75:
                """ 
                3 threads
                """
                print('Starting 3 threads.')
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    executor.map(thread_function,
                                 [os.listdir(dir_path)[0: files_num // 3],
                                  os.listdir(dir_path)[files_num // 3: files_num // 3 * 2],
                                  os.listdir(dir_path)[files_num // 3 * 2: files_num]])
            else:
                if 75 < files_num <= 100:
                    """ 
                    4 threads
                    """
                    print('Starting 4 threads.')
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        executor.map(thread_function,
                                     [os.listdir(dir_path)[0: files_num // 4],
                                      os.listdir(dir_path)[files_num // 4: files_num // 4 * 2],
                                      os.listdir(dir_path)[files_num // 4 * 2: files_num // 4 * 3],
                                      os.listdir(dir_path)[files_num // 4 * 3: files_num]])
                else:
                    """ 
                    1 thread
                    """
                    print('Starting 1 thread.')
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        executor.map(thread_function, [os.listdir(dir_path)])


if __name__ == "__main__":
    #Tagger.tag_dir('./images', False, False)
    Tagger.tag_file("./images/example_01.jpg")
