import os

#from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import format


from numpy.random import seed
seed(42)# keras seed fixing
import tensorflow as tf
tf.random.set_seed(42)# tensorflow seed fixing

EPOCHS = 50
LSTM_NODES = 256
NUM_SENTENCES = 1000 #23700
MAX_SENTENCE_LENGTH = 30
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 300
BATCH_SIZE = 256

input_sentences = []
output_sentences = []
output_sentences_inputs = []

count = 0
for line in open(r'train.txt', encoding="utf-8"):
    count += 1

    if count > NUM_SENTENCES:
        break
    if '\t' in line:
        input_sentence, output = line.rstrip().split('\t')
        output_sentence = output + ' <eos>'
        output_sentence_input = '<sos> ' + output

        input_sentences.append(input_sentence)
        output_sentences.append(output_sentence)
        output_sentences_inputs.append(output_sentence_input)

print("кол-во сэмплов входа", len(input_sentences))
print("кол-во сэмплов выхода:", len(output_sentences))
print("кол-во сэмплов входа-выхода:", len(output_sentences_inputs))

print(input_sentences[172])
print(output_sentences[172])
print(output_sentences_inputs[172])

# TOKENIZATION
from keras.preprocessing.text import Tokenizer
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
print('Всего уникальных слов на входе: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Длина самого длинного предложения на входе: %g" % max_input_len)

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index
print('Всего уникальных слов на выходе: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Длина самого длинного предложения на выходе: %g" % max_out_len)

# PADDING
encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)

decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')



# EMBEDDING
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

embedFile = open(r'ru_embeddings.txt', encoding="utf8")

for line in embedFile:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
embedFile.close()


num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))

for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

print(embeddings_dictionary["поел"])

embedding_layer = Embedding(num_words, EMBEDDING_SIZE,
                            weights=[embedding_matrix], input_length=max_input_len)

decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)

print("(кол-во вхождений, длина предложения на выходе, кол-во слов на выходе)")
print(decoder_targets_one_hot.shape)


decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

def make_model():
    model = Model([encoder_inputs_placeholder,
                   decoder_inputs_placeholder],
                  decoder_outputs)
    return model
model = make_model()

from keras.models import load_model

from keras.callbacks import ModelCheckpoint
choice = ''
'''if os.path.exists('weights.h5'):
    model.load_weights('weights.h5')
    print('Найдены веса. Тренировать (1) или тестировать (2)?')
    choice = input()'''
def compile():
    model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
compile()
'''
callbacks = [
    ModelCheckpoint(
        filepath= 'weights.h5',
        save_weights_only=True,
        save_best_only=True
        )
]'''
#checkpoint = ModelCheckpoint(checkpointPath, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='max')
#callbacks_list = [checkpoint]
from keras.callbacks import Callback

#from keras.utils import plot_model
#plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
model.fit(
        [encoder_input_sequences, decoder_input_sequences],
        decoder_targets_one_hot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        #callbacks = [callback],
        validation_split=0.1
)
#model.save(checkpoint_dir + '/ckpt2')

encoder_model = Model(encoder_inputs_placeholder, encoder_states)

# ------------------------------------------------------

decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

#-------------------------------------

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)
'''
def tryTranslate():
    i = np.random.choice(len(input_sentences))
    input_seq = encoder_input_sequences[i:i + 1]
    translation = translate_sentence(input_seq)
    print('-')
    print('Input:', input_sentences[i])
    print('Response:', translation)'''

def getResponse(inp):
    inp = format.format(inp)
    seq = input_tokenizer.texts_to_sequences(inp)
    pseq = pad_sequences(seq, maxlen=max_input_len)
    response = translate_sentence(pseq)
    return response

from vk_api.longpoll import VkLongPoll, VkEventType
import vk_api
import usrdata

print("Importing userdata from ./usrdata.py")

vk_session = vk_api.VkApi(login=usrdata.login, password=usrdata.password, app_id=2685278)  # kate mobile
vk_session.auth(token_only=True)
session_api = vk_session.get_api()  # получение доступа к API
longpoll = VkLongPoll(vk_session)  # создание longpoll

print("Running")
while True:
    for event in longpoll.listen(): #отслеживание новых сообщений
        if event.type == VkEventType.MESSAGE_NEW: 
            print(event.user_id, ": ", str(event.text))
            if event.text != '':
                userText = event.text.lower()
                if event.from_user and not (event.from_me):
                    r = getResponse(userText)
                    vk_session.method("messages.send", {'user_id': event.user_id, 'message': r, 'random_id': 0})

