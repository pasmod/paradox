import io


class Glove(object):

    @classmethod
    def load(cls, filename):
        dct = {}

        with io.open(filename, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                tokens = line.split(' ')

                word = tokens[0]
                entries = tokens[1:]

                dct[word] = [float(x) for x in entries]
        instance = Glove()
        instance.dct = dct

        return instance

    def vector(self, word):
        return self.dct.get(word, None)
