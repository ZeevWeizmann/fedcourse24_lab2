import torch
class Client:
    r"""Represents a client participating in the learning process

    Attributes
    ----------
    client_id:

    client_id: int

    learner: Learner

    device: str or torch.device

    train_loader: torch.utils.data.DataLoader

    val_loader: torch.utils.data.DataLoader

    test_loader: torch.utils.data.DataLoader

    train_iterator:

    local_steps: int

    metadata: dict

    logger: torch.utils.tensorboard.SummaryWriter

    """
    def __init__(
            self,
            client_id,
            local_steps,
            logger,
            learner=None,
            train_loader=None,
            val_loader=None,
            test_loader=None,
    ):

        self.client_id = client_id

        self.learner = learner

        self.device = self.learner.device

        if train_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

            self.num_samples = len(self.train_loader.dataset)

            self.train_iterator = iter(self.train_loader)

            self.is_ready = True

        else:
            self.is_ready = False

        self.local_steps = local_steps

        self.logger = logger

        self.metadata = dict()

        self.counter = 0

    def step(self, global_model=None, mu=0.0):
        """Perform one local step (FedAvg or FedProx)."""
        self.counter += 1
        self.learner.fit_epochs(
            loader=self.train_loader,
            n_epochs=self.local_steps,
            global_model=global_model,
            mu=mu
        )


    def write_logs(self, counter=None):
        if counter is None:
            counter = self.counter

        train_loss, train_metric = self.learner.evaluate_loader(self.val_loader)
        test_loss, test_metric = self.learner.evaluate_loader(self.test_loader)

        self.logger.add_scalar("Train/Loss", train_loss, counter)
        self.logger.add_scalar("Train/Metric", train_metric, counter)
        self.logger.add_scalar("Test/Loss", test_loss, counter)
        self.logger.add_scalar("Test/Metric", test_metric, counter)
        self.logger.flush()

        return train_loss, train_metric, test_loss, test_metric

class ScaffoldClient(Client):
    """
    Represents a SCAFFOLD client participating in the learning process.
    Maintains local control variates to reduce client drift.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_local = [torch.zeros_like(p.data) for p in self.learner.model.parameters()]

    def step(self, c_global):
        """Perform local updates with control variate correction."""
        self.counter += 1
        self.learner.model.train()
        optimizer = torch.optim.SGD(self.learner.model.parameters(), lr=self.learner.lr)

        model_init = [p.data.clone() for p in self.learner.model.parameters()]

        for _ in range(self.local_steps):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.learner.criterion(self.learner.model(x), y)
                loss.backward()

                #  Correction term
                with torch.no_grad():
                    for p, c_l, c_g in zip(self.learner.model.parameters(), self.c_local, c_global):
                        p.grad += (c_l - c_g)

                optimizer.step()

        # Update local control variates
        with torch.no_grad():
            for idx, param in enumerate(self.learner.model.parameters()):
                delta_w = param.data - model_init[idx]
                self.c_local[idx] = (
                    self.c_local[idx]
                    - c_global[idx]
                    + (1.0 / (self.local_steps * self.learner.lr)) * delta_w
                )
        