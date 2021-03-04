"""
    This is a sample builder script, that retrieves data from several APIs (e.g. WANDB) and data sources
    (e.g. user file) and prettify the information so that it can fill the slots of a HTML template
    representing a prototypical "DAG card". If you run Metaflow with a custom profile,
    remember to set the METAFLOW_PROFILE env variable before running the builder script.

    While the code is fully-functional, it is a very much MVP script, which gets the job done by mixing some
    functionalities found inside Metaflow classes and data from WANDB APIs. Pretty much everything can be
    improved, but the code still does a good job in producing a credible DAG card to receive feedback
    from multiple stakeholders - other ML engineers, PMs, marketing folks, etc.

    For the full back-story, motivations, backlog, please refer to the README file and the
    companion blog post!
"""

# do some imports
from jinja2 import Environment, FileSystemLoader, select_autoescape
# import the Metaflow class containing the target DAG
from training_flow import RegressionModel
import os
import json
from collections import Counter, namedtuple
from datetime import datetime
from metaflow import Flow, Step, FlowSpec, includefile, namespace
from dotenv import load_dotenv
load_dotenv(verbose=True)
# make sure we have an API KEY for wandb in the env variables
assert os.getenv('WANDB_API_KEY') is not None
import wandb

# some global vars
TEMPLATE_FILE_NAME = 'dag_card_template.html'
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CARD_OUTPUT_PATH = 'card'
CARD_VERSION = '0'
DAG_FILE = 'training_flow.py'
# simulate a user db for user metadata: keys are Metaflow users
with open('users.json') as f:
    USER_META = json.load(f)
print("Total of {} users in the user DB!".format(len(USER_META)))
# define a named tuple for data exchange
MetaflowData = namedtuple('MFData', ['top_runs', 'user_counter'])


def get_wandb_data(entity: str, dag_name: str, filter_runs: list = None):
    """
    Retrieve data from WANDB project associated with the DAG, exploiting a naming convention
    between DAG runs and WANDB tracking.

    For the API detailed documentation, see: https://docs.wandb.ai/library/public-api-guide

    :param entity: name of the WANDB entity, i.e. the user running the experiment
    :param dag_name: name of the DAG
    :param filter_runs: if specified, it is the list of runs we want to return (and ignore all not included)
    :return: list of dictionaries, each containing the metrics tracked by WANDB during the specified runs
    """
    api = wandb.Api()
    wandb_project = "{}/{}".format(entity, dag_name)
    print("Getting runs from: {}".format(wandb_project))
    runs = api.runs(wandb_project)
    runs_list = []
    for run in runs:
        if filter_runs and run.name not in filter_runs:
            continue
        # convert pandas containing history info to dicts
        # keys are: '_step', 'loss', '_runtime', 'epoch', '_timestamp', 'val_loss'
        _history = run.history().to_dict()
        history = [{'epoch': e, 'loss': l} for e, l in zip (list(_history['epoch'].values()),
                                                             list(_history['loss'].values()))]
        runs_list.append({
            'name': run.name,
            'history': history
        })

    return runs_list


def find_user_from_tags(run_tags: frozenset):
    """
    For each run, find the user associated with it.

    TODO: is there a better way to get users from MF API?

    :param run_tags: Metaflow tags associated with a run
    :return: user responsible for the run, if any, or None if no user tag is found
    """
    for tag in run_tags:
        if tag.startswith('user:'):
            return tag.replace('user:', '')

    # if no user is found return None (THIS SHOULD NOT HAPPEN, RIGHT?)
    return None


def get_metaflow_runs_and_artifacts(flow_name: str, top_n: int = 2):
    """
    Use Metaflow client to retrieve all runs from all users for a given DAG, and specify how many should be
    returned (for display in card).

    Note that since we are using this function to also compile stats for owners, we first retrieve
    all runs, and then return only the top N.

    For more info on the API, read here: https://docs.metaflow.org/metaflow/client

    :param flow_name: name of the DAG
    :param top_n: max number of runs to displayed in the card
    :return: MetaflowData
    """
    namespace(None)  # -> get all runs from all users
    flow = Flow(flow_name)
    # filter for runs that ended successfully
    runs = [r for r in list(flow) if r.finished]
    print("Total of #{} finished run for {}.".format(len(runs), flow_name))
    # for the latest top n runs, prepare objects to display
    runs_list = []
    for run in runs:
        user = find_user_from_tags(run.tags)
        target_step =  Step('{}/{}/{}'.format(flow_name, run.id, 'join_runs'))
        data = target_step.task.data
        new_run = {
            'user': user,
            'name': run.id,
            'finished_at': run.finished_at,
            # make sure we don't ask for a property in a run without it (e.g. it was added later).
            'model_path': data.best_s3_model_path if 'best_s3_model_path' in data else None,
            # accuracy is the last float in the array
            'accuracy': data.best_model_metrics[-1] if 'best_model_metrics' in data else 0.0,
            'best_learning_rate': data.best_learning_rate if 'best_learning_rate' in data else None,
            'best_model_summary': data.best_model_summary if 'best_model_summary' in data else None
        }
        runs_list.append(new_run)

    # return only top N, but user stats are run on the entire history!
    return MetaflowData(top_runs=runs_list[:top_n], user_counter=Counter(r['user'] for r in runs_list))


def load_jinja_template(path: str, template: str):
    """
    Load jinja template from the file systema.

    :param path: folder containing the HTML templates
    :param template: name of the template
    :return:
    """
    template_loader = FileSystemLoader(searchpath=path)
    env = Environment(loader=template_loader, autoescape=select_autoescape(['html']))

    return env.get_template(template)


def print_dag_chart(dag_file: str, path: str, dag: str):
    """
    Save a local png file depicting the DAG structure, using the built-in
    client method found on: https://github.com/Netflix/metaflow/issues/19

    TODO: investigate a better way to print the info (without running cmd)

    :param dag_file: name of the Python script containing the dag
    :param path: destination folder for the picture
    :param dag: name of the DAG
    :return: name of the pic file
    """
    dag_pic = '{}_{}.png'.format(dag, 'pic')
    cmd = 'python {} output-dot | dot -Tpng -o {}/{}'.format(dag_file, path, dag_pic)
    os.system(cmd)

    return dag_pic


def parse_dag_graph(graph):
    """
    Return a list of steps in the dag, based on the DAG graph object; method inspired by Metaflow
    "show" client command.

    :param graph: Metaflow graph, which is the _graph properties of the DAG class
    :return: list of dictionaries, each one describing a step in the DAG
    """
    steps = []
    for _, node in sorted((n.func_lineno, n) for n in graph):
        steps.append(
            {
                'name': node.name,
                'description': node.doc if node.doc else '?',
                'next': '%s' % ', '.join('*%s*' % n for n in node.out_funcs) if node.name != 'end' else '-'
            }
        )
    print("Total of #{} steps".format(len(steps)))
    return steps


def get_dag_params(obj: FlowSpec):
    """
    Retrieve input files and params in the DAG (for now it assumes the objects are either files or params).

    :param obj: a FlowSpec object from Metaflow
    :return: list of dictionaries, each one describing a file/parameter for the DAG
    """
    params = []
    for p in obj._get_parameters():
        params.append({
            'name': p[0],
            'type': 'file' if isinstance(p[1], includefile.IncludeFile) else 'parameter'
        })

    return params

def build_dag_card():
    """
    Main function, just calling services in turn and compiling the final dictionary
    used by jinja to fill the HTML template.

    :return: None
    """
    # use Metaflow API in a smart way: load but not parse the flow
    obj = RegressionModel(use_cli=False)
    # get flow name
    flow_name = obj.name
    # get the flow dag as a graph
    g = obj._graph
    # now print out dag in a local png file
    dag_pic = print_dag_chart(DAG_FILE, CARD_OUTPUT_PATH, flow_name)
    # get params
    dag_params = get_dag_params(obj)
    # prepare the tasks with verbose text etc.
    steps = parse_dag_graph(g)
    # get metaflow runs
    metaflow_data = get_metaflow_runs_and_artifacts(flow_name=flow_name)
    # get the ids to limit wandb output to only top runs
    run_ids = ['{}:{}-{}'.format(flow_name, r['name'], r['best_learning_rate']) for r in metaflow_data.top_runs]
    # get tracking data from wandb
    wandb_runs = get_wandb_data(entity=os.getenv('WANDB_ENTITY'), dag_name=flow_name, filter_runs=run_ids)
    assert len(metaflow_data.top_runs) == len(run_ids)
    # prepare final list of parameter for HTML slot filling
    params = {
        'model_name': flow_name,
        'card_version': CARD_VERSION,
        'last_update': datetime.date(datetime.now()),
        'model_overview': str(obj.__doc__).replace('\n', ' ').strip(),
        # default to MF user if not user is in the meta db
        'owners': [USER_META.get(user, user) for user in list(metaflow_data.user_counter.keys())],
        # make sure we use the human readable name, when available in the db, for the user chart
        'user_runs': {USER_META.get(user, {}).get('name', user): count
                      for user, count in dict(metaflow_data.user_counter).items()},
        'dag_picture': dag_pic,
        'dag_params': dag_params,
        'flow': steps,
        'metaflow_runs': metaflow_data.top_runs,
        'wandb_runs': wandb_runs
    }
    # get html template for final substitution
    template = load_jinja_template(os.path.join(CURRENT_DIR, 'templates'), TEMPLATE_FILE_NAME)
    # finally, create html page!
    template.stream(params).dump('{}/{}_card.html'.format(CARD_OUTPUT_PATH, flow_name))
    # say bye!
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))
    return


if __name__ == "__main__":
    build_dag_card()