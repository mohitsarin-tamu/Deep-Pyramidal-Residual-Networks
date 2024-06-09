# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
  "num_classes": 10,
  "alpha": 300,
  "depth": 110,
  "weight_decay":  0.0001,
  "batch_size": 128,
  "epochs" : 300,
  "save_interval": 10,
  "modeldir": 'model',
  "learning_rate": 0.1,
  "gamma": 0.1,
  "milestones": [150, 200, 250],
  "clip_norm": 50,
  "momentum": 0.9,
  "inner_lr": 1,
  "inner_iter": 1,
  "lrchange_ep": 120,
  "epsilon": 0.5,
  "best_model_path": '../code/model/model-300.ckpt'
	# ...
}

training_configs = {
		"name": 'MyModel',
	"save_dir": '../saved_models/',
  "num_classes": 10,
  "alpha": 300,
  "depth": 110,
  "weight_decay":  0.0001,
  "batch_size": 128,
  "epochs" : 300,
  "save_interval": 10,
  "modeldir": 'model',
  "learning_rate": 0.1,
  "gamma": 0.1,
  "milestones": [150, 200, 250],
  "clip_norm": 50,
  "momentum": 0.9,
  "inner_lr": 1,
  "inner_iter": 1,
  "lrchange_ep": 120,
  "epsilon": 0.5,
  "best_model_path": '../code/model/model-300.ckpt'
	# ...
}

### END CODE HERE
