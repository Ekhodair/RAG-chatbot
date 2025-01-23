FROM python:3.10.12-slim

LABEL description="Doc QA"

WORKDIR /Doc-QA

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x ./run.sh

EXPOSE 8083
EXPOSE 8501

ENTRYPOINT ["./run.sh"]