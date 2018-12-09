class ClassifyController:
    def __init__(self, result=None):
        self.examinations = [("subject{}".format(i), i) for i in range(20)]
        self.result = result

    # todo classify
    def on_classify_button_click(self, input_data):
        pass

        # self.result.classify_examination = None
        # self.result.classify_examination_length = None
