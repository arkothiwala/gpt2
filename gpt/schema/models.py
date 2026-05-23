from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    seq_len: int
    vocab_size: int
    dropout: float
    activation: str
    position_embedding: str


@dataclass
class OptimizerConfig:
    type: str
    lr: float
    weight_decay: float
    beta1: float
    beta2: float


@dataclass
class SchedulerConfig:
    warmup_steps: int
    schedule: str


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    init_std: float


@dataclass
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig