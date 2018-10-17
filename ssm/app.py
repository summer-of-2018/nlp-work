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
import crfSubprocess
import pandas
import bilsm_crf_model

app = Flask(__name__)
bootstrap = Bootstrap(app)
# bootstrap.init_app(app)
app.config.from_object("config")
app.config['UPLOAD_FOLDER'] = './static/uploads'


@app.route("/", methods=["GET", "POST"])
def search():
    return render_template("pageTable.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    return render_template("predict.html")    


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    return render_template("upload.html")    

# @app.route('/api/uploads', methods = ['GET', 'POST'])
# def uploads():
#     if request.method == 'POST':
#         file = request.files['file_input']
#         filename = secure_filename(file.filename)
#         if file:
#             f, typ = os.path.splitext(filename)
#             if typ == '.xlsx':
#                 inp = handler.excel_extract(file)  # 句子的字符串列表
#                 sentences = [models.cut_sentence(item) for item in inp]
#                 x = models.predict_preprocess(sentences)
#                 tags = models.predict_class(x)  # 预测类别的列表
#                 count_items = crfSubprocess.crf_predict(sentences, tags)  # 计数项字符串列表
#                 genFileName = uuid.uuid4().hex
#                 df = pandas.DataFrame({'count_item': count_items, 'count_class': tags})
#                 df.to_csv(current_app.config['UPLOAD_FOLDER'] + '/' + genFileName, index=True, index_label='id', sep=',')
#                 return jsonify(genFileName)
#             elif typ == '.pdf':
#                 c, res = handler.pdf_extract(file)
#                 with open(current_app.config['UPLOAD_FOLDER'] + '/' + str(time.time()*1000), 'wb') as f:
#                     f.write(c.encode('utf-8'))
#             # file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
#         return jsonify('3e0d0b8a9e094980bbdda4f834ded09a')
#     return render_template('index.html')


@app.route('/api/uploads', methods = ['GET', 'POST'])
def uploads_v2():
    if request.method == 'POST':
        file = request.files['file_input']
        filename = secure_filename(file.filename)
        if file:
            f, typ = os.path.splitext(filename)
            if typ == '.xlsx':
                sentences = handler.excel_extract_v2(file)  # 句子的字符串列表
                tags, count_items = bilsm_crf_model.predict_sentences(sentences)
                genFileName = uuid.uuid4().hex
                df = pandas.DataFrame({'count_item': count_items, 'count_class': tags})
                df.to_csv(current_app.config['UPLOAD_FOLDER'] + '/' + genFileName, index=True, index_label='id', sep=',')
                return jsonify(genFileName)
            elif typ == '.pdf':
                c, res = handler.pdf_extract(file)
                with open(current_app.config['UPLOAD_FOLDER'] + '/' + str(time.time()*1000), 'wb') as f:
                    f.write(c.encode('utf-8'))
            # file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        return jsonify('3e0d0b8a9e094980bbdda4f834ded09a')
    return render_template('index.html')


@app.route("/api/predict", methods=["GET", "POST"])
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

@app.route("/api/getData", methods=["GET", "POST"])
def get_data():
    if request.method == 'POST':
        print(request.json)
        fileName = request.json['fileName']
        if fileName == 'init':
            return jsonify({'total': 0, 'rows': []})

        # try:
        df = pandas.read_csv(current_app.config['UPLOAD_FOLDER'] + '/' + fileName)
        total = int(df.iloc[:,0].size)

        if 'pageSize' in request.json and 'offset' in request.json:
            pageSize = request.json['pageSize']
            offset = request.json['offset']
        else:
            pageSize = total
            offset = 0
        
        sdf = df.iloc[offset:min(offset + pageSize, total-1),]
        rows = sdf.to_dict('records')
        '''
        except:
            total = 0
            rows = []
        '''
        data = {'total': total, 'rows': rows}
        return jsonify(data)
    return jsonify({})

@app.route("/api/edit", methods=["GET", "POST"])
def edit_data():
    if request.method == 'POST':
        print(request.form)
        fileName = request.form['fileName']
        row = []
        row.append(int(request.form['row[id]']))
        row.append(request.form['row[count_class]'])
        row.append(request.form['row[count_item]'])
        df = pandas.read_csv(current_app.config['UPLOAD_FOLDER'] + '/' + fileName)
        df.iloc[row[0], :] = row
        df.to_csv(current_app.config['UPLOAD_FOLDER'] + '/' + fileName, index=False, sep=',')
        return jsonify({})
    return jsonify({})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888)
