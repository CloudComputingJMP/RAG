FROM python:3.13


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install fastapi uvicorn google-cloud-aiplatform vertexai


COPY ./ /code
CMD ["uvicorn","main:app","--host", "0.0.0.0","--port", "8080"]