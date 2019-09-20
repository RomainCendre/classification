from sklearn.base import BaseEstimator, TransformerMixin


class OrderedEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, unknown='Unknown'):
        self.unknown = unknown

    def fit(self, y):
        self.map_list = y
        return self

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        elements = y.tolist()
        if not isinstance(elements, list):
            elements = [elements]
        return np.array([self.__inverse_element(element) for element in elements])

    def transform(self, y):
        elements = y.tolist()
        if not isinstance(elements, list):
            elements = [elements]
        return np.array([self.__transform_element(element) for element in elements])

    def __inverse_element(self, element):
        if element == -1:
            return self.unknown
        else:
            return self.map_list[element]

    def __transform_element(self, element):
        try:
            return self.map_list.index(element)
        except:
            return -1
