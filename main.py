from neuro_core import Neuro
from flask import Flask, request, jsonify
app = Flask('app')


@app.route('/text', methods=['POST'])
def query_text_vectorizer():
    '''
    REST API Method
    json data:
    [
      text: 'text to vectorize'
    }
    :return:
    '''
    json = request.get_json()
    text = json['text']
    print(text)
    vec = neuro.get_vector_for_text(text)
    return jsonify(vec.tolist())


@app.route('/texts', methods=['POST'])
def query_texts_vectorizer():
    '''
    REST API Method
    json data:
    [
      texts: ['text to vectorize', 'text to vectorize', 'text to vectorize', ...]
    }
    :return:
    '''
    json = request.get_json()
    text = json['texts']
    print(text)
    vec = neuro.get_vector_for_texts(text)
    return jsonify(vec.tolist())


if __name__ == '__main__':
    global Neuro
    global neuro
    neuro = Neuro()
    print(neuro.get_vector_for_text('Пример текста'))  # important!
    app.run('0.0.0.0', port=80, debug=False)

