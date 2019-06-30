from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from keras.models import Model
from keras.layers import  Input, Reshape, Bidirectional, LSTM
import scipy.spatial.distance


class Neuro:
    elmo = None
    encoder = None

    def __init__(self):
        self.elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                                 max_token=16)
        print('ELMo nuero core was init')

        self.init_keras_model()
        self.encoder.load_weights('new_gen_3_elmo.model')
        print('encoder neuro core was init')

    def init_keras_model(self):
        # encoder
        inp = Input(shape=(1024,))  # 1024 - размерность
        res1 = Reshape((1, 1024))(inp)
        lstm = Bidirectional(LSTM(128, activation='tanh', input_shape=(1024,)), merge_mode='sum')(res1)
        self.encoder = Model(inp, lstm, name="encoder")
        self.encoder.compile(optimizer='adam', loss='mse')

    def get_vector_for_text(self, text: str):
        elmo_outpt = self.elmo([text.split(' ')])
        return self.encoder.predict(elmo_outpt)[0]

    def get_vector_for_texts(self, texts: [str]):
        elmo_outpt = self.elmo([x.split(' ') for x in texts])
        return self.encoder.predict(elmo_outpt, batch_size=100)


if __name__ == '__main__':
    # tests
    neuro = Neuro()
    print(type(neuro.get_vector_for_text('проверка, как меня слышно?')))
    texts = ['О, новый дивный мир!',
             'Как же тут здорово!',
             'Я думаю, у меня всё получится',
             'У меня всё выйдет',
             'Дивный мир выйдет']
    vecs = neuro.get_vector_for_texts(texts)

    print(' '.join(map(str, [x for x in range(1, len(texts)+1)])))
    for row in range(0, len(texts)):
        print(row+1, end=' ')
        for col in range(0, len(texts)):
            dist = -(scipy.spatial.distance.cosine(vecs[row], vecs[col])-1)
            print(dist, end=' ')
        print()
    '''
    1                    2                  3              4                 5
    1 1.0                0.3454769551753998 0.386410385370 0.362076610326766 0.4681769013404846 
    2 0.3454769551753998 1.0                0.560672760009 0.481747329235076 0.32237571477890015 
    3 0.3864103853702545 0.5606727600097656 1.0            0.896432459354    0.4627530574798584 
    4 0.3620766103267669 0.4817473292350769 0.896432459354 1.0               0.530439019203186 
    5 0.4681769013404846 0.3223757147789001 0.462753057479 0.530439019203186 1.0 
    '''
