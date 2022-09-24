MNIST digits recognizer.

Results:
```
Reading data
60000/60000   Done
10000/10000   Done
Reading data done
Using dense model.
FlattenLayer Layer    in shape:  (*, 28, 28, 1)    out shape: (*, 784)
Dense Layer           in shape:  (*, 784)    out shape: (*, 512)  total params: 401920
Activation Layer      in shape:  (*, 512)    out shape: (*, 512)
DropoutLayer Layer    in shape:  (*, 512)    out shape: (*, 512)  rate: 0.300
Dense Layer           in shape:  (*, 512)    out shape: (*, 512)  total params: 262656
Activation Layer      in shape:  (*, 512)    out shape: (*, 512)
DropoutLayer Layer    in shape:  (*, 512)    out shape: (*, 512)  rate: 0.300
Dense Layer           in shape:  (*, 512)    out shape: (*, 10)  total params: 5130
Activation Layer      in shape:  (*, 10)    out shape: (*, 10)
total params: 669706
   1 [================================] 1:06 train cost: 0.050571 test cost: 0.030399
   2 [================================] 1:07 train cost: 0.024648 test cost: 0.026596
   3 [================================] 1:06 train cost: 0.019629 test cost: 0.021218
   4 [================================] 1:07 train cost: 0.015862 test cost: 0.018637
   5 [================================] 1:06 train cost: 0.014168 test cost: 0.019536
   6 [================================] 1:06 train cost: 0.012497 test cost: 0.018853
   7 [================================] 1:06 train cost: 0.011439 test cost: 0.019084
   8 [================================] 1:06 train cost: 0.010557 test cost: 0.019545

Valid: 9808 / 10000 [98.08%]
```