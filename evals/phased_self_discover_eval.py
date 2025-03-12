import os
import json
import re
import time
import random

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
from tqdm import tqdm
import fire

from self_discover import self_discover
from self_discover.helpers.logger import logger
from helpers.llm import model
from helpers.config import config
from helpers.dataset import load_checkpoints
from helpers.eval import calculate_accuracy