#-*- coding:utf-8 -*-

import subprocess
import re
import os
from threading import Timer

CRF_MODEL = './model/crf_model'
TIEM_SEC = 2000
DECODE = 'utf-8'
TEMP_FILE = 'temp_crf'

def runCRF(input_file):
    cmd = 'crf_test -m {} {}'.format(CRF_MODEL, input_file)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    kill_proc = lambda p: p.kill()
    timer = Timer(TIEM_SEC, kill_proc, [proc])
    try:
        timer.start()
        out, err = proc.communicate()
    finally:
        timer.cancel()
    #proc.wait()
    out = out.decode(DECODE)
    out = out.replace('\r', '')
    sentences = [s for s in out.split('\n\n') if len(s) > 1]
    preds = []
    for st in sentences:
        if '\n' in st:
            lines = st.split('\n')
        else:
            lines = [st]
        p = '-'
        for row in lines:
            cols = row.split('\t')
            if len(cols) < 3:
                #print(cols)
                continue
            if cols[2] == 'T':
                p = cols[0]
                break
        preds.append(p)

    print('%d / %d' % (len(sentences), len(preds)))
    return preds                

def crf_predict(sentences, pred_class):
    # inf is a map sentences and tags
    with open(TEMP_FILE, 'wb') as fp:
        buf = ''
        for i in range(0, len(sentences)):
            st = sentences[i]
            cl = pred_class[i]
            for w in st:
                buf += '{}\t{}\t\n'.format(w, cl)
                # fp.write('{}\t{}\t\n'.format(w, cl), encode='utf-8')
            # fp.write('\n', , encode='utf-8')
            buf += '\n'
        fp.write(buf.encode('utf-8'))

    count_word = runCRF(TEMP_FILE)

    return count_word


if __name__ == '__main__':
    sentences = [['制定', '调查', '问卷', '信息', '，', '包括', '投票', '主题', '、', '时间', '、', '调研', '题目', '等', '信息']]
    pred_class = ['ILF']
    print(crf_predict(sentences, pred_class))