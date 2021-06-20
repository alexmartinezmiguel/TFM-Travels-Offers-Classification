FROM python:3.8

ENV APP_NAME=weather.py

COPY "$APP_NAME" /code/"$APP_NAME"
COPY weather.conf /code/weather.conf
COPY /mapping/functions.py /code/mapping/functions.py

WORKDIR /code

ENV FLASK_APP="$APP_NAME"
ENV FLASK_RUN_HOST=0.0.0.0

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["flask", "run"]
