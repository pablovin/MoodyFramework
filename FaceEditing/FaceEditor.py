# From : Lindt, A., Barros, P., Siqueira, H., & Wermter, S. (2019, May). Facial expression editing with continuous emotion labels. In 2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019) (pp. 1-8). IEEE.
import tensorflow as tf
import numpy
import cv2

class FaceEditor:

    image_size = 96
    image_value_range = [-1, 1]

    def prepareInput(self, image):

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(numpy.float32) * (self.image_value_range[-1] - self.image_value_range[0]) / 255.0 + self.image_value_range[0]

        return numpy.array(image).reshape((1, 96, 96, 3))

    def prepareOutput(self, image):
        image = (image - self.image_value_range[0]) / (self.image_value_range[-1] - self.image_value_range[0])
        image = image * 255

        return image

    def editFaceSequence(self, image, arousals, valences):

        sequence = []

        for index in range(len(arousals)):
            sequence.append(self.editFace(image,arousals[index], valences[index]))

        return sequence


    def editFace(self, image, arousal, valence):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        inputImage = self.prepareInput(image)

        with tf.Session(config=config) as sess:
            with tf.device('/device:CPU:0'):
                # restore graph
                new_saver = tf.train.import_meta_graph('/home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/FaceEditing/checkpoint/01_model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint('./home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/FaceEditing/checkpoint'))
                graph = tf.get_default_graph()

                # create feed dict
                arousal_tensor = graph.get_tensor_by_name("arousal_labels:0")
                valence_tensor = graph.get_tensor_by_name("valence_labels:0")
                images_tensor = graph.get_tensor_by_name("input_images:0")

                query_images = numpy.tile(inputImage, (49, 1, 1, 1))

                # create input for net
                feed_dict = {arousal_tensor: arousal, valence_tensor: valence, images_tensor: query_images}
                op_to_restore = sess.graph.get_tensor_by_name("generator/Tanh:0")

                # run
                x = sess.run(op_to_restore, feed_dict)
                return self.prepareOutput(x[0])