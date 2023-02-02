# Prediction
![](https://github.com/Cyclip/prediction/raw/main/repo/demonstration.png)

Using PyTorch, we can create a prediction model to fit on various functions  
The structure of the neural network is:  
**Layer 1:** 1 neuron (x input)  
**Layer 2:** 4 neurons  
**Layer 3:** 4 neurons  
**Layer 4:** 1 neuron (y output)

## Usage
This is mainly for demonstration and learning purposes, but if you'd like  
to see how this works with other functions, simply look through other  
available functions within `dataGen.py`. Either create your own function  
or pick one you're interested in.  
  
Within `main.py`, change the `DATA_FUNC` constant to the function you desire.  
For example, to change it to the `cos` function within `dataGen.py`, you would  
change the line to `DATA_FUNC = dataGen.cos` within `main.py`.