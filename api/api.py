from flask import Flask, redirect, url_for, render_template, request, flash

app = Flask(__name__, static_url_path='/static')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/verifyMessage', methods=['POST'])
def check_message():
    if request.method == 'POST':
        title = request.form['title']
        subtitle = request.form['subtitle']
        message = request.form['message']
        flash("not test message")
    return redirect(url_for('homepage'))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
