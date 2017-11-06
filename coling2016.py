import csv
import json
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input, Embedding, LSTM, Dense, merge, Reshape, TimeDistributed,Flatten,Dropout,Lambda,Permute
from keras.preprocessing import sequence
from lin_2016.config_pdtb import Sense_To_Label, Label_To_Sense
import numpy as np
from keras.models import Model
from keras.utils import np_utils
import keras.backend as K
from keras.utils.visualize_util import plot
from keras.optimizers import Adagrad
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from gensim.models import Word2Vec


try:
    import cPickle as pickle
except ImportError:
    import pickle

np.random.seed(1337)

def TMP_2level(s):
    z = s.split(".")
    if len(z)>2:
        z = z[:2]
    return ".".join(z)

class precessing():
    def __init__(self):
        self.file = "output_section/"
        self.train_file = self.file + "02-20.json"
        self.dev_file = self.file + "00-01.json"
        self.test_file = self.file + "21-22.json"
        self.w2v_file = "../../GoogleNews-vectors-negative300.bin"
        # self.w2v_file = "vec.txt"
        self.word_len = 20
        self.arg_len = 80
        self.w2v_dims = 300


    def arg_process(self, file):
        fo = open(file)
        relation = [json.loads(x) for x in fo]
        fo.close()
        data = []
        for r in relation:
            if r["Type"] != "Implicit":
                continue
            temp = {}
            temp["Arg1"] = r["Arg1"]["RawText"].split()
            temp["Arg2"] = r["Arg2"]["RawText"].split()
            temp["Sense"] = r["Sense"]
            data.append(temp)
        return data

    def ch_to_index(self, text, tok=None):
        sequences = []
        if tok is None:
            tokenizer = Tokenizer(lower=False, char_level=True)
            all_of_them = [' '.join(z) for z in text]
            tokenizer.fit_on_texts(all_of_them)
        else:
            tokenizer = tok
        for words in text:
            charaters = []
            for ch in tokenizer.texts_to_sequences_generator(words):
                charaters.append(ch)
            sequences.append(charaters)
        return sequences, tokenizer

    def word_to_index(self, text, tok=None):
        real_text = [' '.join(z) for z in text]
        if tok is None:
            tokenizer = Tokenizer(lower=False, filters=" ")
            tokenizer.fit_on_texts(real_text)
        else:
            tokenizer = tok
        # here do not need the loop, just put the list of sentences (str) as input
        sequences = tokenizer.texts_to_sequences(real_text)
        # tokenizer.word_docs.items()
        return sequences, tokenizer

    def vocab(self,file):
        data = self.arg_process(file)
        arg1 = []
        arg2 = []
        for x in data:
            for s in x["Sense"]:
                arg1.append(x["Arg1"])
                arg2.append(x["Arg2"])
        with open("args.txt", "ab+") as f:
            pickle.dump(arg1+arg2, f)

    def data_prepare(self, file, iftrain):
        data = self.arg_process(file)
        arg1 = []
        arg2 = []
        sense = []
        allx = []
        for x in data:
            if iftrain:
                for s in x["Sense"]:
                    s = TMP_2level(s)
                    if s in Sense_To_Label:
                        arg1.append(x["Arg1"])
                        arg2.append(x["Arg2"])
                        sense.append(Sense_To_Label[s])
                        allx.append(x)
            else:
                s = x['Sense'][0]
                s = TMP_2level(s)
                if s in Sense_To_Label:
                    arg1.append(x["Arg1"])
                    arg2.append(x["Arg2"])
                    sense.append(Sense_To_Label[s])
                    allx.append(x)

        "arg words -> ch index -> get word_docs"
        tok = None
        if not iftrain:
            with open("ch_docs.txt", "rb") as f:
                tok = pickle.load(f)
        arg_ch, tok = self.ch_to_index(arg1 + arg2, tok)
        if iftrain:
            with open("ch_docs.txt", "wb") as f:
                pickle.dump(tok, f)
        arg1_ch = arg_ch[:len(arg_ch) // 2]
        arg2_ch = arg_ch[len(arg_ch) // 2:]

        "arg words -> word index -> get word_docs"
        word_tok = None
        if not iftrain:
            with open("word_docs.txt", "rb") as f:
                word_tok = pickle.load(f)
        arg_word, word_tok = self.word_to_index(arg1 + arg2,word_tok)
        if iftrain:
            with open("word_docs.txt", "wb") as f:
                pickle.dump(word_tok, f)
        arg1_word = arg_word[:len(arg_word) // 2]
        arg2_word = arg_word[len(arg_word) // 2:]

        " padding for words"
        "arg1 ch->word padding"
        temp = []
        for s in arg1_ch:
            temp.append(sequence.pad_sequences(s, maxlen=self.word_len, padding='post', truncating='post'))
        arg1_ch = temp
        "arg2 ch->word padding"
        temp = []
        for s in arg2_ch:
            temp.append(sequence.pad_sequences(s, maxlen=self.word_len, padding='post', truncating='post'))
        arg2_ch = temp

        # padding for sentences
        X_ch_1 = sequence.pad_sequences(arg1_ch, maxlen=self.arg_len, padding='pre', truncating='pre')
        X_ch_2 = sequence.pad_sequences(arg2_ch, maxlen=self.arg_len, padding='post', truncating='post')
        X_word_1 = sequence.pad_sequences(arg1_word, maxlen=self.arg_len, padding='pre', truncating='pre')
        X_word_2 = sequence.pad_sequences(arg2_word, maxlen=self.arg_len, padding='post', truncating='post')
        y = np_utils.to_categorical(np.array(sense))
        return X_ch_1, X_ch_2, X_word_1, X_word_2, y, allx

    def w2v_process(self):
        with open("word_docs.txt", "rb") as f:
            word_tok = pickle.load(f)
        word_docs = word_tok.word_index
        print("Vob-size: ", len(word_tok.word_index))
        WE = np.zeros((45000,self.w2v_dims),dtype='float32')
        wv = Word2Vec.load_word2vec_format(self.w2v_file,binary=True)
        pre_trained = set(wv.vocab.keys())
        for x in word_docs:
            if x in pre_trained:
                WE[word_docs[x],:] = wv[x]
            else:
                WE[word_docs[x],:] = np.array(np.random.uniform(-0.5/self.w2v_dims,0.5/self.w2v_dims,(self.w2v_dims,)),dtype='float32')   #hyperparameter
        return WE

class train_model():
    def __init__(self, train_data, dev_data, test_data,WE,ch_filter_length=None, ch_nb_filter=100,
                 word_filter_length=None, word_nb_filter=100,lstm_size=256,lr=0.001,dense_num=1,dense_size=512,dropout=0.3,batch_size=64):
        self.word_filter_length = word_filter_length
        self.word_nb_filter = word_nb_filter
        self.ch_filter_length = ch_filter_length
        self.ch_nb_filter = ch_nb_filter
        self.ch_dim = 90
        self.word_dim = 45000
        self.ch_ndims = 30
        self.word_ndims = 300
        self.word_maxlen = 20
        self.arg_maxlen = 80
        self.activation = "tanh"
        self.mask = True
        self.lr = lr
        self.epoch = 40
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.dense_num = dense_num
        self.dense_size = dense_size
        self.WE = WE
        self.dropout = dropout

    def lstmCh_word_pro(self):

        ''' input '''
        arg1_ch_input = Input(shape=(self.arg_maxlen * self.word_maxlen,), dtype='int32', name='arg1_ch')
        arg2_ch_input = Input(shape=(self.arg_maxlen * self.word_maxlen,), dtype='int32', name='arg2_ch')
        arg1_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg1_word')
        arg2_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg2_word')
        arg1_mask = K.not_equal(arg1_word_input, 0)
        arg2_mask = K.not_equal(arg2_word_input, 0)

        ''' embedding'''
        # chareters
        emb_ch = Embedding(input_dim=self.ch_dim, input_length=self.arg_maxlen * self.word_maxlen,
                           output_dim=self.ch_ndims)
        arg1_ch = emb_ch(arg1_ch_input)
        arg2_ch = emb_ch(arg2_ch_input)
        arg1_ch = Reshape((self.arg_maxlen, self.word_maxlen, self.ch_ndims))(arg1_ch)
        arg2_ch = Reshape((self.arg_maxlen, self.word_maxlen, self.ch_ndims))(arg2_ch)

        # words
        emb_word = Embedding(input_dim=self.word_dim, input_length=self.arg_maxlen, weights=[self.WE],output_dim=self.word_ndims,trainable=False)
        arg1_word = emb_word(arg1_word_input)
        arg2_word = emb_word(arg2_word_input)


        ''' convolution ch'''
        ch_conv1 = Convolution1D(nb_filter=self.ch_nb_filter,
                             filter_length=self.ch_filter_length[0],
                             border_mode='same',
                             activation=self.activation,
                             subsample_length=1)
        ch_conv2 = Convolution1D(nb_filter=self.ch_nb_filter,
                     filter_length=self.ch_filter_length[1],
                     border_mode='same',
                     activation=self.activation,
                     subsample_length=1)
        ch_conv3 = Convolution1D(nb_filter=self.ch_nb_filter,
             filter_length=self.ch_filter_length[2],
             border_mode='same',
             activation=self.activation,
             subsample_length=1)

        ''' pooling ch '''
        mpool = MaxPooling1D(pool_length=self.word_maxlen)
        tdcnn1 = TimeDistributed(ch_conv1)
        tdcnn2 = TimeDistributed(ch_conv2)
        tdcnn3 = TimeDistributed(ch_conv3)
        tdpol = TimeDistributed(mpool)

        '''arg1 cnn+pooling'''
        arg1_tconv1 = tdcnn1(arg1_ch)
        arg1_tpool1 = tdpol(arg1_tconv1)
        arg1_tpool_res1 = Reshape((self.arg_maxlen, self.ch_nb_filter))(arg1_tpool1)
        arg1_tconv2 = tdcnn2(arg1_ch)
        arg1_tpool2 = tdpol(arg1_tconv2)
        arg1_tpool_res2 = Reshape((self.arg_maxlen, self.ch_nb_filter))(arg1_tpool2)
        arg1_tconv3 = tdcnn3(arg1_ch)
        arg1_tpool3 = tdpol(arg1_tconv3)
        arg1_tpool_res3 = Reshape((self.arg_maxlen, self.ch_nb_filter))(arg1_tpool3)

        '''arg2 cnn+pooling'''
        arg2_tconv1 = tdcnn1(arg2_ch)
        arg2_tpool1 = tdpol(arg2_tconv1)
        arg2_tpool_res1 = Reshape((self.arg_maxlen, self.ch_nb_filter))(arg2_tpool1)
        arg2_tconv2 = tdcnn2(arg2_ch)
        arg2_tpool2 = tdpol(arg2_tconv2)
        arg2_tpool_res2 = Reshape((self.arg_maxlen, self.ch_nb_filter))(arg2_tpool2)
        arg2_tconv3 = tdcnn3(arg2_ch)
        arg2_tpool3 = tdpol(arg2_tconv3)
        arg2_tpool_res3 = Reshape((self.arg_maxlen, self.ch_nb_filter))(arg2_tpool3)

        '''lstm'''
        arg1_ch_cnn_merge = merge([arg1_tpool_res1,arg1_tpool_res2,arg1_tpool_res3], mode='concat', concat_axis=2)
        arg2_ch_cnn_merge = merge([arg2_tpool_res1,arg2_tpool_res2,arg2_tpool_res3], mode='concat', concat_axis=2)

        # arg1_lstmf = LSTM(self.lstm_size,return_sequences=True)(arg1_ch_cnn_merge, mask=arg1_mask)
        # arg1_lstmb = LSTM(self.lstm_size,go_backwards=True,return_sequences=True)(arg1_ch_cnn_merge, mask=arg1_mask)
        # arg2_lstmf = LSTM(self.lstm_size,return_sequences=True)(arg2_ch_cnn_merge, mask=arg2_mask)
        # arg2_lstmb = LSTM(self.lstm_size,go_backwards=True,return_sequences=True)(arg2_ch_cnn_merge, mask=arg2_mask)
        arg1_lstmf = LSTM(self.lstm_size,return_sequences=True)(arg1_ch_cnn_merge)
        arg1_lstmb = LSTM(self.lstm_size,go_backwards=True,return_sequences=True)(arg1_ch_cnn_merge)
        arg2_lstmf = LSTM(self.lstm_size,return_sequences=True)(arg2_ch_cnn_merge)
        arg2_lstmb = LSTM(self.lstm_size,go_backwards=True,return_sequences=True)(arg2_ch_cnn_merge)
        arg1_lstm = merge([arg1_lstmf,arg1_lstmb,arg1_word],mode='concat')
        arg2_lstm = merge([arg2_lstmf,arg2_lstmb,arg2_word],mode='concat')

        arg1_lstm = Dropout(self.dropout)(arg1_lstm)
        arg2_lstm = Dropout(self.dropout)(arg2_lstm)

        ''' ch+word-level cnn + pooling'''
        word_cnn1 = Convolution1D(nb_filter=self.word_nb_filter,
                                 filter_length=self.word_filter_length[0],
                                 border_mode='same',
                                 activation=self.activation,
                                 subsample_length=1)
        word_cnn2 = Convolution1D(nb_filter=self.word_nb_filter,
                                 filter_length=self.word_filter_length[1],
                                 border_mode='same',
                                 activation=self.activation,
                                 subsample_length=1)
        word_cnn3 = Convolution1D(nb_filter=self.word_nb_filter,
                                 filter_length=self.word_filter_length[2],
                                 border_mode='same',
                                 activation=self.activation,
                                 subsample_length=1)

        '''11-02 day do -------change the pooling value for second cnn------'''
        word_mpol = MaxPooling1D(pool_length=2)

        arg1_word_cnn1 = word_cnn1(arg1_lstm)
        arg1_word_cnn2 = word_cnn2(arg1_lstm)
        arg1_word_cnn3 = word_cnn3(arg1_lstm)
        arg2_word_cnn1 = word_cnn1(arg2_lstm)
        arg2_word_cnn2 = word_cnn2(arg2_lstm)
        arg2_word_cnn3 = word_cnn3(arg2_lstm)

        # merge all cnn arg:
        arg1_cnn_merge = merge([arg1_word_cnn1,arg1_word_cnn2,arg1_word_cnn3])
        arg2_cnn_merge = merge([arg2_word_cnn1,arg2_word_cnn2,arg2_word_cnn3])
        arg1_word_mp = word_mpol(arg1_cnn_merge)
        arg2_word_mp = word_mpol(arg2_cnn_merge)

        flatten = Flatten()
        arg1_word_mp = flatten(arg1_word_mp)
        arg2_word_mp = flatten(arg2_word_mp)

        merged_vector = merge([arg1_word_mp,arg2_word_mp], mode='concat', concat_axis=-1)

        for i in range(0,self.dense_num):
            merged_vector = Dense(self.dense_size,activation=self.activation)(merged_vector)
            print("add %d times dense..."%(i))

        merged_vector = Dropout(self.dropout)(merged_vector)

        predictions = Dense(11, activation='softmax', name="output")(merged_vector)


        model = Model(input=[arg1_ch_input, arg2_ch_input, arg1_word_input, arg2_word_input],
                      output=predictions)
        # model.summary()
        # plot(model, to_file='lstmcnn.png')

        ada = Adagrad(lr=self.lr, epsilon=1e-06)
        model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def fit(self, model, filename):
        '''data = [X_ch_1,X_ch_2,X_word_1,X_word_2,y]'''
        print(filename,self.epoch,self.lr,self.dense_num,self.dense_size,"ch_filter_length",self.ch_filter_length,"ch_nb_filter",self.ch_nb_filter,
              "word_filter_length",self.word_filter_length,"word_nb_filter",self.word_nb_filter)
        train = self.train_data
        dev = self.dev_data
        test = self.test_data
        stop = EarlyStopping(patience=5, verbose=1,monitor='val_acc',mode='max')
        mcheck = ModelCheckpoint("%s-model"%(filename),monitor='val_acc',verbose=1,save_best_only=True,mode='max')
        hist = model.fit({'arg1_ch': self.reshape_arg(train[0]), 'arg2_ch': self.reshape_arg(train[1]),
                          'arg1_word': train[2], 'arg2_word': train[3]}, train[4], nb_epoch=self.epoch, validation_data= \
                             ({'arg1_ch': self.reshape_arg(dev[0]), 'arg2_ch': self.reshape_arg(dev[1]),
                               'arg1_word': dev[2], 'arg2_word': dev[3]}, dev[4]), callbacks=[stop,mcheck])
        val_acc = max(hist.history['val_acc'])

        model.load_weights("%s-model"%(filename))
        test_acc = model.evaluate({'arg1_ch': self.reshape_arg(test[0]), 'arg2_ch': self.reshape_arg(test[1]),
                                 'arg1_word': test[2], 'arg2_word': test[3]}, test[4], batch_size=self.batch_size)
        stri = "%s, val_acc: %f, test_acc %f,self.dense_num %f,self.dense_size %f, ,self.lstmsize %f,ch_filter_length1: %d,ch_filter_length2: %d,ch_filter_length3: %d, ch_nb_filter: %d, " \
               "word_filter_length1: %d,word_filter_length2: %d,word_filter_length3: %d, word_nb_filter: %d, lr: %f, batch_size: %d." % \
               (filename, val_acc, test_acc[1], self.dense_num,self.dense_size,self.lstm_size,self.ch_filter_length[0],self.ch_filter_length[1],self.ch_filter_length[2], self.ch_nb_filter,
                self.word_filter_length[0],self.word_filter_length[1],self.word_filter_length[2], self.word_nb_filter,self.lr,self.batch_size)+"\r\n"
        str2 = "|   |"+str(val_acc)+"  "+str(test_acc)+"  "+filename
        print(stri)
        print(str2)

        ''' special test '''
        pred_pr_return = model.predict({'arg1_ch': self.reshape_arg(test[0]), 'arg2_ch': self.reshape_arg(test[1]),
                                 'arg1_word': test[2], 'arg2_word': test[3]})
        pred_pr = pred_pr_return
        pred_y = [np.argmax(z) for z in pred_pr]
        print("Real test acc: ", train_model.get_acc(pred_y, test[5]))

        ''' write record '''

        csvfile = open('final_tacl_dev.csv', 'a+')
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow([filename, train_model.get_acc(pred_y, test[5]),val_acc, test_acc[1],self.lr,self.dense_num,self.dense_size,self.ch_filter_length[0],self.ch_filter_length[1],self.ch_filter_length[2],
                         self.ch_nb_filter,self.word_filter_length[0],self.word_filter_length[1],self.word_filter_length[2], self.word_nb_filter,self.batch_size,self.dropout])




    def reshape_arg(self, arg):
        X = np.reshape(arg, (arg.shape[0], arg.shape[1] * arg.shape[2]))
        return X

    @staticmethod
    def get_acc(pred_y, allx):
        assert len(pred_y) == len(allx)
        num_all = 0
        num_right = 0
        for pi, g in zip(pred_y, allx):
            num_all += 1
            TMP_senses = [TMP_2level(i) for i in g["Sense"]]
            if Label_To_Sense[pi] in TMP_senses:
                num_right += 1
        return (num_right+0.) / num_all


if __name__ == '__main__':
    p = precessing()
    train_data = p.data_prepare(p.train_file, True)
    dev_data = p.data_prepare(p.dev_file, False)
    test_data = p.data_prepare(p.test_file, False)
    WE = p.w2v_process()
    cn_nb_filter,word_nb_filter,ch_filter_length,word_filter_length = 128,1024,(2,3,4),(2,4,8)
    for i in range(10):
        try:
            model = train_model(train_data, dev_data, test_data, WE,
                                ch_filter_length=ch_filter_length, ch_nb_filter=cn_nb_filter,
                                word_filter_length=word_filter_length, word_nb_filter=word_nb_filter,
                                lstm_size=50,dense_num=1,dense_size=100,dropout=0.25,batch_size=128,lr=0.002)
            model.fit(model.lstmCh_word_pro(),"lstmcnn_final_tacl")
            # model.fit(model.word_represention(),"word_represention")
        except Exception as e:
            print (Exception,":",e)
