import sys
import os
import random
import json
import time
from collections import defaultdict
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.seed import seed_everything
from src.data.preprocessing import create_df
from src.utils.label_mapping_regplans import label_to_id
from llm_stuff.evaluation import evaluate

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (SystemMessage, HumanMessage)
from langchain_core.exceptions import OutputParserException
from openai import RateLimitError, BadRequestError

from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rouge_score import rouge_scorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

base_dir = Path(os.getcwd())
config = load_config(base_dir / 'secrets.yaml')

seed_everything(42)

global val_df
val_df = create_df(base_dir / 'data/my_data/regplans-dev.conllu')

os.environ['OPENAI_API_VERSION'] = config['OPENAI_API_VERSION']
os.environ['AZURE_OPENAI_ENDPOINT'] = config['OPENAI_API_BASE']
os.environ['AZURE_OPENAI_API_KEY'] = config['AZURE_OPENAI_API_KEY']

llm = AzureChatOpenAI(
    deployment_name=config['OPENAI_DEPLOYMENT_NAME'],
    temperature=0.0
)

with open(base_dir / 'llm_stuff/prompts/examples.json', 'r') as f:
    example_bank = json.load(f)

def format_examples(example_subset):
    # Formats the examples into a string for later prompt
    formatted = []
    for i, ex in enumerate(example_subset):
        entity_lines = "\n".join([f"{e['word']} {e['label']}" for e in ex["entities"]])
        formatted.append(
            f"Eksempel {i+1}:\n"
            f"Tekst: \"{ex['sentence']}\"\n"
            f"Entiteter:\n{entity_lines}\n##\n"
        )
    return "\n".join(formatted)

def evaluate_example_subset(examples, sentence, tokens, true_labels):
    formatted_examples = format_examples(examples)

    msg = [
        SystemMessage(
            content=(
                "Du er en ekspert på Named Entity Recognition (NER). Din oppgave er å identifisere entiteter "
                "som representerer feltnavn i tekstutdrag fra reguleringsplaner."
            )
        ),
        HumanMessage(
            content=f"""\
De eneste gyldige etikettene er B-FELT (begynnelsen på et feltnavn) og I-FELT (fortsettelsen av det samme feltnavnet).

{formatted_examples}

Formuler svaret over flere linjer, med ett token per linje, og kun tokens som inngår i ett feltnavn. Hver linje skal inneholde tokenet etterfulgt av tilhørende etikett, atskilt med ett mellomrom.

Tekst: '{sentence}'

Entiteter:
"""
        )
    ]

    max_retries = 5
    retry_delay = 10  # Initial delay in seconds

    response = None  # Default

    for attempt in range(max_retries):
        try:
            response = llm.invoke(msg)
            break
        except BadRequestError as e:
            print(f"BadRequestError: {e}. Skipping this prompt.")
            return 0.0
        except ValueError as e:
            print(f"ValueError (possibly content filter): {e}. Skipping this prompt.")
            return 0.0
        except (RateLimitError, OutputParserException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retryable error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

    if response is None or not hasattr(response, "content"):
        print("Response was None or invalid. Skipping.")
        return 0.0

    entities = defaultdict(list) # Word-label pairs

    for line in response.content.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            word, label = parts[0], parts[1]
            entities[word].append(label)

    pred_labels = []
    word_counts = defaultdict(int)  # Track occurrences of each word

    for token in tokens:
        if token in entities and word_counts[token] < len(entities[token]):
            pred_labels.append(entities[token][word_counts[token]])  # Get the label in order
            word_counts[token] += 1  # Increment occurrence counter
        else:
            pred_labels.append("O")  # Default to "O" if missing

    pred_ids = []
    for label in pred_labels:
        if label in label_to_id:
            pred_ids.append(label_to_id[label])
        else:
            pred_ids.append(label_to_id.get("O", -1))

    true_ids = [label_to_id[label] for label in true_labels]
    metrics = evaluate(true_ids, pred_ids)

    return metrics['f1'] # Return f1 score

NUM_EXAMPLES = len(example_bank)
SUBSET_SIZE = 5
POP_SIZE = 10
NUM_GEN = 20
CXPB = 0.5
MUTPB = 0.3
TOURNSIZE = 2

ga_params = {
    "population_size": POP_SIZE,
    "num_generations": NUM_GEN,
    "crossover_probability": CXPB,
    "mutation_probability": MUTPB,
    "selection_tournament_size": TOURNSIZE,
    "subset_size": SUBSET_SIZE
}

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # try to maximize the f1
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_sample", lambda: random.sample(range(NUM_EXAMPLES), SUBSET_SIZE))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_sample)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutate(individual):
    idx_to_replace = random.randint(0, SUBSET_SIZE - 1)
    available_examples = list(set(range(NUM_EXAMPLES)) - set(individual)) # Get examples not in the subset
    if available_examples:
        new_example = random.choice(available_examples)
        individual[idx_to_replace] = new_example
    print("After mutation:", individual)
    return (individual,)

def evaluate_fitness(individual):
    examples = [example_bank[i] for i in individual]
    scores = []

    global val_df
    val_df_sample = val_df.iloc[:int(len(val_df) * 0.5)] 
    
    for _, row in val_df_sample.iterrows():
        sentence = row['full_text']
        tokens = row['words']
        true_labels = row['labels']
        score = evaluate_example_subset(examples, sentence, tokens, true_labels)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    return (avg_score,) # Return the average f1 score

toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop = toolbox.population(n=POP_SIZE)
pop, logbook = algorithms.eaSimple(
    pop, toolbox,
    cxpb=CXPB,
    mutpb=MUTPB,
    ngen=NUM_GEN,
    stats=stats,
    verbose=True
)

best_individual = tools.selBest(pop, 1)[0]
best_examples = [example_bank[i] for i in best_individual]
print("Best example subset:", best_examples)

gen = logbook.select("gen")
avg = logbook.select("avg")
std = logbook.select("std")
min_ = logbook.select("min")
max_ = logbook.select("max")

log_data = {
    "parameters": ga_params,
    "logbook": {
        "gen": gen,
        "avg": avg,
        "std": std,
        "min": min_,
        "max": max_
    }
}

with open("genetic_algorithm_results.json", "w") as f:
    json.dump(log_data, f, indent=4)

plt.plot(gen, avg, label='avg')
plt.fill_between(gen, np.array(avg) - np.array(std), np.array(avg) + np.array(std), alpha=0.2)
plt.plot(gen, min_, label='min')
plt.plot(gen, max_, label='max')
plt.legend()
plt.savefig("genetic_algorithm.png")
plt.show()
