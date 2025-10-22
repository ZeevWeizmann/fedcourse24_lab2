"""Train

This script allows to train one federated learning experiment; the dataset name, the algorithm and the
number of clients should be precised alongside with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * train - training class ready for federated learning simulation

"""

from utils.args import *
from utils.utils import *

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_):

    clients = []

    data_dir = get_data_dir(args_.experiment)

    for client_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, client_dir)) and client_dir.startswith('client_'):

            client_id = int(client_dir.replace('client_', ''))
            client_dir = os.path.join(data_dir, f"client_{client_id}")
            logs_dir = os.path.join(args_.logs_dir, f"client_{client_id}")
            os.makedirs(logs_dir, exist_ok=True)
            logger = SummaryWriter(logs_dir)

            client = \
                init_client(
                    args=args_,
                    client_id=client_id,
                    client_dir=client_dir,
                    logger=logger
                )

            clients.append(client)

    return clients


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    seed = (args.seed if (("seed" in args) and (args.seed >= 0)) else int(time.time()))
    torch.manual_seed(seed)

    print("==> Initialize Clients..")
    
    clients = \
        init_clients(args_=args)

    clients_weights = get_clients_weights(
        clients=clients,
        objective_type=args.objective_type,
    )
    if getattr(args, "algorithm", None) not in ["fedprox", "scaffold", None]:
         raise ValueError(f"Unknown algorithm: {args.algorithm}")
    print("==> Initialize Aggregator..")
    
    global_learner = \
        get_learner(
            experiment_name=args.experiment,
            device=args.device,
            optimizer_name=args.server_optimizer,
            lr=args.server_lr,
            mu=args.mu,
            seed=seed
        )

    global_logs_dir = os.path.join(args.logs_dir, "global")
    os.makedirs(global_logs_dir, exist_ok=True)
    global_logger = SummaryWriter(global_logs_dir)

    if getattr(args, "algorithm", None) == "scaffold":
        from aggregator import ScaffoldAggregator
        aggregator = ScaffoldAggregator(
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=global_logger,
            sampling_rate=args.sampling_rate,
            sample_with_replacement=args.sample_with_replacement,
            verbose=args.verbose,
            seed=args.seed,
        )
    elif getattr(args, "algorithm", None) == "fedprox":
        from aggregator import FedProxAggregator
        aggregator = FedProxAggregator(
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=global_logger,
            sampling_rate=args.sampling_rate,
            sample_with_replacement=args.sample_with_replacement,
            verbose=args.verbose,
            seed=args.seed,
            mu=args.mu
        )
    else:
        aggregator = \
        get_aggregator(
            aggregator_type=args.aggregator_type,
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=global_logger,
            sampling_rate=args.sampling_rate,
            sample_with_replacement=args.sample_with_replacement,
            verbose=args.verbose,
            seed=seed
        )

    print("==> Training Loop..")

    aggregator.write_logs()

    # Training loop
    for c_round in tqdm(range(args.n_rounds)):

        # Perform a training round by calling aggregator.mix()
        aggregator.mix()

        # Write logs
        if (c_round % args.log_freq) == (args.log_freq - 1):
            aggregator.write_logs()
