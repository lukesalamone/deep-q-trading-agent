from pipelines.transfer_learning import gather_groups
from pipelines.run_numq import train
from models.models import *
from typing import List, Dict

def load_weights(model: DQN, IN_PATH):
    model.policy_net.load_state_dict(torch.load(IN_PATH))
    model.hard_update()
    return model

def save_weights(model: DQN, OUT_PATH):
    torch.save(model.target_net.state_dict(), OUT_PATH)
    return

def pretrain_on_group(profits: Dict, groups: Dict, index:str, method: str, group:str):
    for i, symbol in enumerate(groups[index][method][group]):
        model = DQN(NUMQ)
        if i > 0:
            previous_weights = f'weights/numq/{index}/{method}/{group}/numq_{symbol}_{i}.pt'
            model = load_weights(model=model, IN_PATH=previous_weights)
        model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                          index=index,
                                                                          symbol=symbol,
                                                                          episodes=10,
                                                                          dataset='train')
        current_weights = f'weights/numq/{index}/{method}/{group}/numq_{symbol}_{i+1}.pt'
        save_weights(model=model, OUT_PATH=current_weights)

    return profits

def evaluate_groups(groups: Dict, index:str, method: str, group:str):
    # for i, symbol in enumerate(groups[index][method][group]):
    # TODO: EVALUATE LAST MODEL IN EACH GROUP
    #  SELECT BEST MODEL ACCORDING TO TOTAL VALID PROFIT
    pass

if __name__=='__main__':
    groups = gather_groups()
    print(groups)