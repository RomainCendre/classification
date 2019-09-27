import keras
from keras.layers import Dense
from keras import applications
from keras import Sequential, Model
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from toolbox.core.layers import RandomLayer
from toolbox.core.models import KerasBatchClassifier


class Applications:

    @staticmethod
    def get_application(architecture='InceptionV3', pooling='max'):
        # We get the deep extractor part as include_top is false
        if architecture == 'VGG16':
            model = applications.VGG16(weights='imagenet', include_top=False, pooling=pooling)
        elif architecture == 'InceptionResNetV2':
            model = applications.InceptionResNetV2(weights='imagenet', include_top=False, pooling=pooling)
        elif architecture == 'NASNet':
            model = applications.NASNetLarge(weights='imagenet', include_top=False, pooling=pooling)
        elif architecture == 'ResNet':
            model = applications.ResNet50(weights='imagenet', include_top=False, pooling=pooling)
        else:
            model = applications.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
        model.name = architecture

        return model

    @staticmethod
    def get_preprocessing_application(architecture='InceptionV3'):
        # We get the deep extractor part as include_top is false
        if architecture == 'VGG16':
            return applications.vgg16.preprocess_input
        elif architecture == 'InceptionResNetV2':
            return applications.inception_resnet_v2.preprocess_input
        elif architecture == 'NASNet':
            return applications.nasnet.preprocess_input
        elif architecture == 'ResNet':
            return applications.resnet50.preprocess_input
        else:
            return applications.inception_v3.preprocess_input

    @staticmethod
    def get_transfer_tuning(architecture='InceptionV3'):
        extractor_params = {'architecture': architecture,
                            'pooling': 'avg',
                            'preprocessing_function': Applications.get_preprocessing_application(architecture=architecture)}
        return KerasBatchClassifier(Applications.get_application, **extractor_params)


    @staticmethod
    def get_fine_tuning(output_classes, architecture='InceptionV3'):

        # We get the deep extractor part as include_top is false
        base_model = Applications.get_application(architecture, pooling='avg')

        x = base_model.output
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(output_classes, activation='softmax')(x)

        return Model(inputs=base_model.inputs, outputs=predictions)

    @staticmethod
    def get_dummy_deep(output_classes):
        keras.layers.RandomLayer = RandomLayer
        # Extract labels
        model = Sequential()
        model.add(RandomLayer(output_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.name = 'Dummy_Deep'
        return model

    @staticmethod
    def get_dummy_simple():
        pipe = Pipeline([('clf', DummyClassifier())])
        pipe.name = 'Dummy_Simple'
        # Define parameters to validate through grid CV
        parameters = {}
        return pipe, parameters