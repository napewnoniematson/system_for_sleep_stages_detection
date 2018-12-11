from src.logger import colors
from src.utils.util import *


# todo plot figure results
class ResultController:
    def __init__(self, result=None):
        self.result = result

    def update(self):
        return '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}'.format(
            self.highlight_title(TRAINING_TEXT),
            self.one_line(MODEL_NAME_TEXT, self.result.model_name),
            self.one_line(EXAMINATIONS_TEXT, self.list_str(self.result.examinations)),
            self.one_line(FEATURES_TEXT, self.list_str([f.__name__ for f in self.result.features]
                                                       if self.result.features else None)),
            self.one_line(WINDOW_WIDTH_TEXT, self.result.window_width),
            self.one_line(STAGE_MODE_TEXT, StageMode(self.result.stage_mode).name if self.result.stage_mode else None),
            self.one_line(EPOCHS_TEXT, self.result.epochs),
            self.one_line(TRAINING_PART_TEXT, self.result.training_part),
            self.one_line(ACCURACY_TEXT, self.result.accuracy),
            self.highlight_title(CLASSIFY_TEXT),
            self.one_line(EXAMINATIONS_FOR_CLASSIFY_TEXT, self.result.classify_examination),
            self.one_line(EXAMINATIONS_FOR_CLASSIFY_LENGTH_TEXT, self.result.classify_examination_length)
        )

    def highlight_title(self, title):
        return "*** {} ***\n".format(title)

    def one_line(self, text_word, value):
        return text_word + ": " + str(value) + '\n'

    def list_str(self, l):
        return ', '.join(l) if l else 'None'
