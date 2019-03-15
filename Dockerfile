FROM python:3.6

ADD requirements.txt requirements.txt

ADD api/ /

RUN pip install -r requirements.txt

CMD [ "python", "./main.py" ]