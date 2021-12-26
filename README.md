## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. main.py - This contains code for our Machine Learning model to predict that  transaction is fraud or legit absed on training data in 'creditcard.csv' file.
2. app.py - This contains Flask APIs that receives input through GUI or API calls, computes the precited value based on our model and returns it either 0(not a fraud) or 1(fraud).
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter he input accoding to dataset i.e 29 inputs 

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python main.py
```
This would create a serialized version of our model into a file credit_fraud.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage.

Enter valid numerical/float values in all 29 input boxes and hit <b>START ANALYSIS</b>.

If everything goes well, you should  be able to see the predcited vaule(either 0 or 1) on the HTML page!

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
thank you for visiting
