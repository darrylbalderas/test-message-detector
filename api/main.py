from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/verifyMessage', methods=['POST'])
def check_message():
    if request.method == 'POST':
        if request.form['title'] == 'wca':
            return redirect(url_for('message_status', status="test"))
        return redirect(url_for('message_status', status="notest"))
    return redirect(url_for('homepage'))


@app.route('/messageStatus/<status>', methods=['GET'])
def message_status(status):
    if status == "test":
        return redirect(url_for('test'))
    return redirect(url_for('no_test'))


@app.route('/test')
def test():
    return 'Test message'


@app.route('/notest')
def no_test():
    return 'Not a test message'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
