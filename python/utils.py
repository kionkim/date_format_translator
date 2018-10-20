import pickle
import mxnet as mx
import numpy as np
import pandas as pd
from datetime import datetime
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
from mxnet.ndarray.linalg import gemm2


def generate_date_data(N, in_seq_len = 32, out_seq_len = 32):
    N_train = int(N * .9)
    N_validation = N - N_train
    
    added = set()
    questions = []
    answers = []
    answers_y = []
    
    while len(questions) < N:
        a = gen_date()
        if a in added:
            continue
        question = '[{}]'.format(a)
        answer = '[' + str(format_date(a)) + ']'
        answer = padding(answer, out_seq_len)
        answer_y = str(format_date(a)) + ']'
        answer_y = padding(answer_y, out_seq_len)

        added.add(a)
        questions.append(question)
        answers.append(answer)
        answers_y.append(answer_y)

    # Check the first 20000 characters to build vocab
    chars = list(set(''.join(questions[:20000])))
    chars.extend(['[', ']']) # Start and End of Expression
    chars.extend(list(set(''.join(answers[:20000]))))
    chars = np.sort(list(set(chars)))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    X = np.zeros((len(questions), in_seq_len, len(chars)), dtype=np.integer)
    Y = np.zeros((len(questions), out_seq_len, len(chars)), dtype=np.integer)
    Z = np.zeros((len(questions), out_seq_len, len(chars)), dtype=np.integer)

    for i in range(N):
        for t, char in enumerate(questions[i]):
            X[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers[i]):
            Y[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers_y[i]):
            Z[i, t, char_indices[char]] = 1
    return X, Y, Z, chars, char_indices, indices_char

def gen_test(N):
    q = []
    y = []
    
    for i in range(N):
        question = gen_date()
        answer_y = format_date(question)
        q.append(question)
        y.append(answer_y)
    return(q,y)

def gen_date():
    rnd = int(np.random.uniform(low = 1000000000, high = 1350000000))
    timestamp = datetime.fromtimestamp(rnd)
    return str(timestamp.strftime('%Y-%B-%d %a')) # '%Y-%B-%d %H:%M:%S'

def format_date(x):
    return str(datetime.strptime(x, '%Y-%B-%d %a').strftime('%m/%d/%Y, %A')).lower() #'%H%M%S:%Y%m%d'


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

    
def padding(chars, maxlen):
    if len(chars) < maxlen:
        return chars + ' ' * (maxlen - len(chars))
    else:
        return chars[:maxlen]
    
    
def calculate_loss(model, data_iter, loss_obj, ctx = mx.cpu()):
    test_loss = []
    for i, (x_data, y_data, z_data) in enumerate(data_iter):
        x_data = x_data.as_in_context(ctx).astype('float32')
        y_data = y_data.as_in_context(ctx).astype('float32')
        z_data = z_data.as_in_context(ctx).astype('float32')
        with autograd.predict_mode():
            z_output = model(x_data, y_data)
            loss_te = loss_obj(z_output, z_data)
        curr_loss = nd.mean(loss_te).asscalar()
        test_loss.append(curr_loss)
    return np.mean(test_loss)


def train(model, tr_data_iterator, te_data_iterator, trainer, loss, char_indices, indices_char, epochs  = 10, ctx = mx.cpu(), output_file_name = '../python/result'):
    tot_test_loss = []
    tot_train_loss = []
    for e in range(epochs):
        train_loss = []
        for i, (x_data, y_data, z_data) in enumerate(tr_data_iterator):
            x_data = x_data.as_in_context(ctx).astype('float32')
            y_data = y_data.as_in_context(ctx).astype('float32')
            z_data = z_data.as_in_context(ctx).astype('float32')

            with autograd.record():
                z_output = model(x_data, y_data)
                loss_ = loss(z_output, z_data)
            loss_.backward()
            trainer.step(x_data.shape[0])
            curr_loss = nd.mean(loss_).asscalar()
            train_loss.append(curr_loss)

        if e % 10 == 0:
            q, y = gen_test(10)
            n_correct = 0
            for i in range(10):
                with autograd.predict_mode():
                    p, attn = model.predict(q[i], char_indices, indices_char, input_digits = x_data.shape[1], lchars = len(indices_char))
                    p = p.strip()
                    iscorr = 1 if p == y[i] else 0
                    if iscorr == 1:
                        print(colors.ok + '☑' + colors.close, end=' ')
                        n_correct += 1
                    else:
                        print(colors.fail + '☒' + colors.close, end=' ')
                    print("{} = {}({}) {}".format(q[i], p, y[i], str(iscorr)))
                if n_correct == 10:
                    #file_name = "format_translator.params"
                    #model.save_parameters(file_name)
                    with open('{}_{}.pkl'.format(output_file_name, e), 'wb') as picklefile:
                        pickle.dump(model, picklefile)
                        
        #caculate test loss
        test_loss = calculate_loss(model, te_data_iterator, loss_obj = loss, ctx=ctx) 
        print("Epoch %s. Train Loss: %s, Test Loss : %s" % (e, np.mean(train_loss), test_loss))    
        tot_test_loss.append(test_loss)
        tot_train_loss.append(np.mean(train_loss))
    return tot_test_loss, tot_train_loss

def plot_attention(model, data, char_indices, indices_char, in_seq_len):
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    p =[]
    attn = []
    for i, d  in enumerate(data):
        _p, _attn = model.predict(d, char_indices, indices_char, input_digits = in_seq_len, lchars = len(char_indices))
        p.append(_p.strip())
        attn.append(_attn)

    fig, axes = plt.subplots(np.int(np.ceil(len(data) / 1)), 1, sharex = False, sharey = False)
    plt.subplots_adjust(hspace=1)

    if len(data) > 1:
        fig.set_size_inches(5, 40)
    else:
        fig.set_size_inches(10, 10)
    plt.subplots_adjust(hspace=1)
    
    for i, (d, p, a) in enumerate(zip(data, p, attn)):
        _col = list(d)
        _idx = list(p)
        _val = a[:len(p), :len(d)]
        print('input: {}, length: {}'.format(d,len(d)))
        print('prediction: {}, length:{}'.format(p,len(p)))
        print('attention shape= {}'.format(a.shape))
        print('check attn = {}'.format(np.sum(a, axis = 1)))
        print('val shape= {}'.format(_val.shape))
        if len(data) > 1:
            sns.heatmap(pd.DataFrame(_val, index = _idx, columns = _col), ax = axes.flat[i], cmap = 'RdYlGn', linewidths = .3)
        else:
            sns.heatmap(pd.DataFrame(_val, index = _idx, columns = _col), cmap = 'RdYlGn', linewidths = .3)
        #axes.flat[i].set_title('Label: {}, Pred: {}'.format(_label[i], np.int(_pred[i])))
    pass