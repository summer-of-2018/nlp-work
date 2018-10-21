from flask import Flask, render_template, jsonify, request, current_app
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
import os
import json
import handler
import time
import codecs
import uuid
import models
# models = []
import crfSubprocess
import pandas

# app.config['UPLOAD_FOLDER'] = './static/uploads'


def val():
    inp = []
    y = []
    with open('实验/summary_2.txt', 'r', encoding='utf-8', errors='ignore') as inf:
        for line in inf.readlines():
            line = line.strip()
            print("line:", line)
            if line:
                t_vec = line.split('\t')
                if len(t_vec) >= 3:
                    dec = t_vec[0]
                    dec = dec.strip()
                    inp.append(dec)
                    count_name = t_vec[1]
                    label = t_vec[2]
                    y.append(label)
    sentences = [models.cut_sentence(item) for item in inp]
    x = models.predict_preprocess(sentences)
    tags = models.predict_class(x)
    count_items = crfSubprocess.crf_predict(sentences, tags)
    count = 0
    for i in range(len(y)):
        if y[i] == tags[i]:
            count += 1
    print(count)
    print(len(y))


def uploads():
    if request.method == 'POST':
        file = request.files['file_input']
        filename = secure_filename(file.filename)
        if file:
            f, typ = os.path.splitext(filename)
            if typ == '.xlsx':
                inp = handler.excel_extract(file)
                sentences = [models.cut_sentence(item) for item in inp]
                x = models.predict_preprocess(sentences)
                tags = models.predict_class(x)
                count_items = crfSubprocess.crf_predict(sentences, tags)
                genFileName = uuid.uuid4().hex
                df = pandas.DataFrame({'count_item': count_items, 'count_class': tags})
                df.to_csv(current_app.config['UPLOAD_FOLDER'] + '/' + genFileName, index=True, index_label='id',
                          sep=',')
                return jsonify(genFileName)
            elif typ == '.pdf':
                c, res = handler.pdf_extract(file)
                with open(current_app.config['UPLOAD_FOLDER'] + '/' + str(time.time() * 1000), 'wb') as f:
                    f.write(c.encode('utf-8'))
            # file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        return jsonify('3e0d0b8a9e094980bbdda4f834ded09a')
    return render_template('index.html')


# @app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    '''
    def test():
        data = {}
        data['input'] = '["HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US"]'
        r = requests.post('http://127.0.0.1:5000/api/predict', data=data)
        print r.text
    '''
    if request.method == 'POST':
        print(request.form['input'])
        inp = request.form['input'].strip()
        sentence = []
        sentence.append(models.cut_sentence(inp))
        print(sentence)
        x = models.predict_preprocess(sentence)
        pred = models.predict_class(x)
        print(pred)
        res = {'sentence': sentence, 'predict': pred[0]}
        return jsonify(res)
    return render_template('index.html')


if __name__ == "__main__":
    val()
