import json
import sys
sys.path.append("..")
import experimenter
from experimenter import data, training


def test_classification():
    config = json.load(open("classification.json", 'r'))
    trainer = training.BasicTrainer(config)
    config = trainer()
    print(config)
    
def test_classification_arabic():
    config = json.load(open("classification_arabic.json", 'r'))
    trainer = training.BasicTrainer(config)
    config = trainer()
    print(config)

def test_lm():
    # Test model runs on dummy data
    # Test model achieves 0 loss on training data
    config = json.load(open("lm.json", 'r'))
    trainer = training.BasicTrainer(config)
    config = trainer()
    print(config)

def test_pair():
    config = json.load(open("pairclassification.json", 'r'))
    trainer = training.BasicTrainer(config)
    config = trainer()
    print(config)

def test_multi():
    config = json.load(open("multipairlm.json", 'r'))
    trainer = training.BasicTrainer(config)
    config = trainer()
    print(config)
