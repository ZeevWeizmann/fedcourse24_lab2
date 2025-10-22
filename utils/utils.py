from models import *
from learner import *

from datasets.mnist import *

from client import *

from aggregator import *

from .optim import *
from .metrics import *
from .constants import *

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize


def experiment_not_implemented_message(experiment_name):
    error = f"{experiment_name} is not available! " \
            f"Possible are: 'mnist'."

    return error


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment
    :param experiment_name: name of the experiment
    :return: str
    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_model(experiment_name, device):
    """
    create model

    Parameters
    ----------
    experiment_name: str

    device: str
        either cpu or cuda


    Returns
    -------
        model (torch.nn.Module)

    """

    if experiment_name == "mnist":
        model = LinearLayer(input_dim=784, output_dim=10, bias=True)
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = model.to(device)

    return model


def get_learner(experiment_name, device, optimizer_name, lr, mu, seed):
    """
    constructs learner for an experiment for a given seed

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist"}

    device: str
        used device; possible `cpu` and `cuda`

    optimizer_name: str

    lr: float
        learning rate

    mu: float
        proximal term weight, only used when `optimizer_name=="prox_sgd"`

    seed: int

    Returns
    -------
        Learner

    """
    torch.manual_seed(seed)

    if experiment_name == "mnist":
        criterion = nn.CrossEntropyLoss().to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = \
        get_model(experiment_name=experiment_name, device=device)

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr=lr,
            mu=mu
        )

    if optimizer_name == "prox_sgd":
        learner_class = FedProxLearner
    else:
        learner_class = Learner

    return learner_class(
         model=model,
         criterion=criterion,
         metric=metric,
         device=device,
         optimizer=optimizer,
         is_binary_classification=is_binary_classification,
         lr=lr)


def get_loader(experiment_name, client_data_path, batch_size, train):
    """

    Parameters
    ----------
    experiment_name: str

    client_data_path: str

    batch_size: int

    train: bool

    Returns
    -------
        * torch.utils.data.DataLoader

    """

    if experiment_name == "mnist":
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNIST(root=client_data_path, train=train, transform=transform)

    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def init_client(args, client_id, client_dir, logger):
    """initialize one client


    Parameters
    ----------
    args:

    client_id: int

    client_dir: str

    logger:

    Returns
    -------
        * Client

    """
    train_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=client_dir,
        batch_size=args.bz,
        train=True,
    )

    val_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=client_dir,
        batch_size=args.bz,
        train=False,
    )

    test_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=client_dir,
        batch_size=args.bz,
        train=False,
    )

    learner = \
        get_learner(
            experiment_name=args.experiment,
            device=args.device,
            optimizer_name=args.local_optimizer,
            lr=args.local_lr,
            mu=args.mu,
            seed=args.seed
        )

    if getattr(args, "algorithm", "fedavg") == "scaffold":
         client = ScaffoldClient(
            client_id=client_id,
            learner=learner,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            local_steps=args.local_steps,
            logger=logger
         )
    else:
            client = Client(
                client_id=client_id,
                learner=learner,
                train_loader=train_loader,
                 val_loader=val_loader,
                test_loader=test_loader,
                local_steps=args.local_steps,
                logger=logger
            )

    return client


def get_aggregator(
        aggregator_type,
        clients,
        clients_weights,
        global_learner,
        logger,
        sampling_rate,
        sample_with_replacement,
        verbose,
        seed
):
    """
    Parameters
    ----------
    aggregator_type: str
        possible are {"centralized", "no_communication"}

    clients: Dict[int: Client]

    clients_weights: Dict[int: Client]

    global_learner: Learner

    logger: torch.utils.tensorboard.SummaryWriter

    sampling_rate: float

    sample_with_replacement: bool

    verbose: int

    seed: int


    Returns
    -------
        * Aggregator
    """
    if aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            verbose=verbose,
            seed=seed,
        )
    else:
        error_message = f"{aggregator_type} is not a possible aggregator type, possible are: "
        for type_ in AGGREGATOR_TYPES:
            error_message += f" {type_}"


def get_clients_weights(clients, objective_type):
    """Compute the weights to be associated with every client.

    If objective_type is "average", clients receive the same weight.
    If objective_type is "weighted", clients receive weight proportional to the number of samples.

    Parameters
    ----------
    clients: List[Client]
    objective_type: str
        Type of the objective function; possible are: {"average", "weighted"}

    Returns
    -------
    clients_weights: List[float]
    """
    n_clients = len(clients)
    clients_weights = []

    total_num_samples = 0
    for client in clients:
        total_num_samples += client.num_samples

    for client in clients:

        if objective_type == "average":
            weight = 1 / n_clients

        elif objective_type == "weighted":
            weight = client.num_samples / total_num_samples

        else:
            raise NotImplementedError(
                f"{objective_type} is not an available objective type. Possible are 'average' and 'weighted'.")

        clients_weights.append(weight)

    return clients_weights
