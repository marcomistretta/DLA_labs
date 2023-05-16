from datetime import datetime
from Parser import Parser
from TrainerRL import TrainerRL


current_time = datetime.now().strftime('%H_%M_%S_%d_%m_%y')
# xxx nota bene, per semplicità non ho messo method in Parser, ma lo ho lasciato qui
method = "ppo"  # xxx method can be "dqln" or "ppo"
name = method + "/" + method + "_" + current_time

writer_path = "runs/" + name
save_path = 'checkpoints/' + name + '.pth'
load_path = 'checkpoints/' + name + '.pth'

args = Parser().args
trainer = TrainerRL(method, writer_path, args)

try:
    if args.TRAIN:
        trainer.train(save_path)
        trainer.test(load_path)
    else:
        trainer.test(load_path)
finally:
    trainer.save(save_path)