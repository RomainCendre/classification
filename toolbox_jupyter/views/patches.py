from pathlib import Path
from PIL import Image, ImageDraw

class Patchs:

    def __init__(self, inputs, settings):
        self.inputs = inputs
        self.settings = settings

    def write_patch(self, output_folder):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        output_folder = output_folder / self.inputs.name
        output_folder.mkdir(exist_ok=True)

        references = list(set(self.inputs.get_from_key('Reference')))

        for index, reference in enumerate(references):
            work_input = self.inputs.sub_inputs({'Reference': [reference]})
            path = list(set(work_input.get_from_key('Full_path')))
            label = list(set(work_input.get_from_key('Label')))
            image = Image.open(path[0]).convert('RGBA')
            for sub_index, entity in work_input.data.iterrows():
                start = entity['Patch_Start']
                end = entity['Patch_End']
                center = ((end[0] + start[0]) / 2, (end[1] + start[1]) / 2)
                center = tuple(np.subtract(center, 10)), tuple(np.add(center, 10))
                predict = entity['PredictorTransform']
                color = self.settings.get_color(self.inputs.decode('label', predict)) + (0.5,)  # Add alpha
                color = tuple(np.multiply(color, 255).astype(int))
                draw = ImageDraw.Draw(image)
                draw.rectangle(center, fill=color)
                # draw.rectangle((start, end), outline="white")
            image.save(output_folder / '{ref}_{lab}.png'.format(ref=reference, lab=label[0]))