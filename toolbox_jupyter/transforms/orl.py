

class Preprocessing:

    @staticmethod
    def apply_average_filter(inputs, tags, size):
        """This method allow user to apply an average filter of 'size'.

        Args:
            size: The size of average window.

        """
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        inputs[tags['datum']] = inputs[tags['datum']].apply(lambda x: np.correlate(x, np.ones(size) / size, mode='same'))
        return inputs


    @staticmethod
    def apply_scaling(inputs, tags, method='default'):
        """This method allow to normalize spectra.

            Args:
                method(:obj:'str') : The kind of method of scaling ('default', 'max', 'minmax' or 'robust')
            """
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        if method == 'max':
            inputs[tags['datum']] = inputs[tags['datum']].apply(lambda x: preprocessing.maxabs_scale(x))
        elif method == 'minmax':
            inputs[tags['datum']] = inputs[tags['datum']].apply(lambda x: preprocessing.minmax_scale(x))
        elif method == 'robust':
            inputs[tags['datum']] = inputs[tags['datum']].apply(lambda x: preprocessing.robust_scale(x))
        else:
            inputs[tags['datum']] = inputs[tags['datum']].apply(lambda x: preprocessing.scale(x))
        return inputs




    @staticmethod
    def integrate(inputs, tags):
        self.data['datum'] = self.data.apply(lambda x: [0, np.trapz(x['datum'], x['wavelength'])], axis=1)


    @staticmethod
    def norm_patient_by_healthy(self):
        query = self.to_query(self.filters)
        if query:
            data = self.data.query(query)
        else:
            data = self.data

        for name, group in data.groupby(self.tags['group']):
            # Get features by group
            row_ref = group[group.label == 'Sain']
            if len(row_ref) == 0:
                data.drop(group.index)
                continue
            mean = np.mean(row_ref.iloc[0]['datum'])
            std = np.std(row_ref.iloc[0]['datum'])
            for index, current in group.iterrows():
                data.iat[index, data.columns.get_loc('datum')] = (current['datum'] - mean) / std


    @staticmethod
    def norm_patient(self):
        query = self.to_query(self.filters)
        if query:
            data = self.data.query(query)
        else:
            data = self.data

        for name, group in data.groupby(self.tags['group']):
            # Get features by group
            group_data = np.array([current['datum'] for index, current in group.iterrows()])
            mean = np.mean(group_data)
            std = np.std(group_data)
            for index, current in group.iterrows():
                data.iat[index, data.columns.get_loc('datum')] = (current['datum'] - mean) / std

    @staticmethod
    def ratios(self):
        for name, current in self.data.iterrows():
            wavelength = current['wavelength']
            data_1 = current['datum'][np.logical_and(540 < wavelength, wavelength < 550)]
            data_2 = current['datum'][np.logical_and(570 < wavelength, wavelength < 580)]
            data_1 = np.mean(data_1)
            data_2 = np.mean(data_2)
            self.data.iloc[name, self.data.columns.get_loc('datum')] = data_1 / data_2