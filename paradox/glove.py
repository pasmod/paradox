import io


def map_dim_to_file(dim=200):
    if dim == 50:
        return "glove.6B/glove.6B.50d.txt"
    elif dim == 100:
        return "glove.6B/glove.6B.100d.txt"
    elif dim == 200:
        return "glove.6B/glove.6B.200d.txt"
    elif dim == 300:
        return "glove.6B/glove.6B.300d.txt"
    else:
        return ValueError("Dimension {} is not supported!".format(dim))


class Glove(object):

    @classmethod
    def load(cls, dim=200):
        filename = map_dim_to_file(dim=dim)
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
