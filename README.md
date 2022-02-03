- ## ABOUT
This is a project on credit card fraud detection which is deployed on production using Flask API
***
- ## PREREQUISITES
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.
***
- ## PROJECT STRUCTURE
  ### This project has four major parts :
  - #### main.py 
     This contains code for our Machine Learning model to predict that  transaction is fraud or legit based on training data        in 'creditcard.csv' file.
  - #### app.py 
     This contains Flask APIs that receives input through GUI or API calls, computes the precited value based on our model and      returns it either 0(not a fraud) or 1(fraud).
  - #### request.py 
     This uses requests module to call APIs already defined in app.py and dispalys the returned value.
  - #### templates 
     This folder contains the HTML template to allow user to enter he input accoding to dataset i.e 29 inputs 
***
- ## RUNNING THE PROJECT
    - ### :-
      - Ensure that you are in the project home directory.
      - Create the machine learning model by running below command 
           ```
           python main.py
           ```
      - This would create a serialized version of our model into a file **`credit_fraud.pkl`**

    - ### Run app.py using below command to start Flask API
        ```
        python app.py
        ```
      - By default, flask will run on port 5000.

    - ### :-
      -  Navigate to URL http://localhost:5000
      - You should be able to view the homepage.
      - Enter valid numerical/float values in all 29 input boxes and hit `<b>START ANALYSIS</b>`.
      - If everything goes well, you should  be able to see the predcited vaule(either 0 or 1) on the HTML page!

    - ### You can also send direct POST requests to FLask API using Python's inbuilt request module
      - Run the beow command to send the request with some pre-popuated values -
          ```
          python request.py
          ```
***
- ## THANK-YOU FOR VISITING 💯
