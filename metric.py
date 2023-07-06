class MatchMetirc(object):
    def __init__(self):
        self.cnt = 0
        self.correct = 0

    def update(self, gts, preds):
        for gt, pred in zip(gts, preds):
            if gt == pred:
                self.correct += 1

            self.cnt += 1

    def compute(self):
        return self.correct/self.cnt

    def reset(self):
        self.cnt, self.correct = 0, 0