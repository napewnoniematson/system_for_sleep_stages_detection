class ClassifyController:
    def __init__(self):
        self.examinations = [("subject{}".format(i), i) for i in range(20)]

    def on_classify_button_click(self):
        print('classify')
