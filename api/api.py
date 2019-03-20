from flask import Flask, redirect, url_for, render_template, request, flash
import sys
sys.path.insert(0, './preprocess')
import ml_model as ml
import preprocess_text as pt

app = Flask(__name__, static_url_path='/static')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/verifyMessage', methods=['POST'])
def check_message():
    if request.method == 'POST':
        title = [check_empty(pt.clean_sentence(request.form['title']))]
        subtitle = [check_empty(pt.clean_sentence(request.form['subtitle']))]
        message = [check_empty(pt.clean_sentence(request.form['message']))]
        clf, cvec = ml.load_ml_tools()
        input = ml.convert_fields(title, subtitle, message, cvec)
        if clf.predict(input)[0] == 1:
            flash("test message")
        else:
            flash("not test message")
    return redirect(url_for('homepage'))


def check_empty(parameter):
    if parameter.strip() == "":
        return "emptySt"
    return parameter


if __name__ == '__main__':
    app.run(host='0.0.0.0')
