from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser, PDFDocument
import pandas as pd
import re

def pdf_extract(in_file):
    content = ''
    res = ''
    parser_pdf = PDFParser(in_file)
    doc = PDFDocument()
    
    parser_pdf.set_document(doc)
    doc.set_parser(parser_pdf)
    
    doc.initialize()
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # 创建PDf资源管理器 来管理共享资源
        rsrcmgr = PDFResourceManager()
        
        # 创建一个PDF参数分析器
        laparams = LAParams()
        
        # 创建聚合器
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        
        # 创建一个PDF页面解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        # ot = doc.get_outlines()
        

        # 循环遍历列表，每次处理一页的内容
        # doc.get_pages() 获取page列表
        for page in doc.get_pages():
            # 使用页面解释器来读取
            interpreter.process_page(page)

            # 使用聚合器获取内容
            layout = device.get_result()

            # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等 想要获取文本就获得对象的text属性，
            for out in layout:
                # 判断是否含有get_text()方法，图片之类的就没有
                # if hasattr(out,"get_text"):
                if isinstance(out, LTTextBoxHorizontal):
                    if res == '':
                        res = out.get_text()
                    results = out.get_text()
                    content += results
    return content, res


def excel_extract(in_file):
    inp = []
    try:
        df = pd.read_excel(in_file, usecols=[7])
        # print(df)
        for index,row in df.iterrows():
            if pd.notnull(row['功能单元描述']):
                sentence = row['功能单元描述'].strip()
                sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！“”，。：？、~@#￥%……&*（）]+", "", sentence)
                inp.append(sentence)
    except Exception as e:
        print('open file faild!')
        print(e)

    return inp


if __name__ == '__main__':
    fp = open('D:\MLpro\FunctionRecognition\data\\2\项目33.xlsx', 'rb')
    excel_extract(fp)