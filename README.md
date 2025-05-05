# Taxi-v3 Reinforcement Learning

## How to Use

To train and evaluate Q-Learning algorithm, run: 

``` bash
python main.py --algo q_learning --mode all
```


To train and evaluate SARSA algorithm, run: 

``` bash
python main.py --algo sarsa --mode all
```


To only train the model, run:

``` bash
python main.py --algo sarsa --mode evaluate
```


To only evaluate the model, run:

``` bash
python main.py --algo sarsa --mode train --episodes 5000
```


The results and plots will be automatically logged in the `results` folder.