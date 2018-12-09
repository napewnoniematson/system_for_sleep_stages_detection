class ClassifyController:
    def __init__(self, result=None):
        self.examinations = [("subject{}".format(i), i) for i in range(20)]
        self.result = result

    def on_classify_button_click(self, input_data):
        print(self.result.window_width)
        print(self.result.stage_mode)
        print(self.result.epochs)
        # self.result.classify_examination = None
        # self.result.classify_examination_length = None
