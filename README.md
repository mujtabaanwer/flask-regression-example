# Flask Regression Example

## How to Use Without Docker

### Install the depdendencies
`pip install -r requirements.txt`

### Run the Flask App
`python mainapp.py`

## How to use With Docker

all recipe / step are described in `Dockerfile` files

## Build the Docker Images

`docker build -t myregression:latest .`

## Run the Docker Images

`docker run --name regression -p 5000:5000 myregression:latest`


## How to Retraining the data
`python train.py`

this will read from file `./dataset/DummySalaryData.csv`  and dump it as trained model located at `./trained` directory

## sample test data

using `testdata.csv` feel free to edit the values

## Using The App

Use the Docker / not will be have same result

open the browser at `http://localhost:5000` and upload the csv file's