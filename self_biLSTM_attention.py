import os

# model 순서
# /home/minwookje/coding/cbaziotis/ntua-slp-semeval2018/modules/nn/models.py
ModelWrapper(
    # /home/minwookje/coding/cbaziotis/ntua-slp-semeval2018/modules/nn/models.py
    FeatureExtractor(
        Embed(
            Embedding(804871,310)
            Dropout(0.1)
            GaussianNoise(mean=0.0, stddev=0.2)
        )
        # /home/minwookje/coding/cbaziotis/ntua-slp-semeval2018/modules/nn/modules.py
        RNNEncoder(
            LSTM(310, 250, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
            Dropout(p=0.3)
        )
        SelfAttention(
            Sequential(
                Linear(in_features=500, out_features=1, bias=True)
                Tahh()
                Dropout(0.3)
                Linear(in_features=500, out_features=1, bias=True)
                Tanh()
                Dropout(p=0.3)
            )
            Softmax()
        )
    )
    Linear(in_features=500, out_features=11, bias=True)    
)

###########################[1]전역 변수
#  전역변수 (1. path )

# ???????Training.py 에서 trainner class가져오기

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TRAINED_PATH = os.path.join(BASE_PATH, "trained")

EXPS_PATH = os.path.join(BASE_PATH, "out/experiments")

ATT_PATH = os.path.join(BASE_PATH, "out/attentions")

DATA_DIR = os.path.join(BASE_PATH, 'datasets')

#  전역변수 (2. config(hyper parameter))

TASK1_EC = {
    "name": "TASK1_E-c",
    "token_type": "word",
    "batch_train": 32,
    "batch_eval": 32,
   "epochs": 50,
    #"embeddings_file": "word2vec_300_6_concatened",
    "embeddings_file": "ntua_twitter_affect_310",	
    "embed_dim": 310,
    "embed_finetune": False,
    "embed_noise": 0.2,
    "embed_dropout": 0.1,
    "encoder_dropout": 0.3,
    "encoder_size": 250,
    "encoder_layers": 2,
    "encoder_bidirectional": True,
    "attention": True,
    "attention_layers": 2,
    "attention_context": False,
    "attention_activation": "tanh",
    "attention_dropout": 0.3,
    "base": 0.56,
    "patience": 20,
    "weight_decay": 0.0,
    "clip_norm": 1,
}

############################## [2]함수 정의


# 데이터 load하기하기
def load_datasets(datasets, train_batch_size, eval_batch_size, token_type,
                  preprocessor=None,
                  params=None, word2idx=None, label_transformer=None):
    if params is not None:
        name = "_".join(params) if isinstance(params, list) else params
    else:
        name = None

    loaders = {}
    if token_type == "word":
        if word2idx is None:
            raise ValueError

        if preprocessor is None:
            preprocessor = twitter_preprocess()

        print("Building word-level datasets...")
        for k, v in datasets.items():
            _name = "{}_{}".format(name, k)
            dataset = WordDataset(v[0], v[1], word2idx, name=_name,
                                  preprocess=preprocessor,
                                  label_transformer=label_transformer)
            batch_size = train_batch_size if k == "train" else eval_batch_size
            loaders[k] = DataLoader(dataset, batch_size, shuffle=True,
                                    drop_last=True)

    elif token_type == "char":
        print("Building char-level datasets...")
        for k, v in datasets.items():
            _name = "{}_{}".format(name, k)
            dataset = CharDataset(v[0], v[1], name=_name,
                                  label_transformer=label_transformer)
            batch_size = train_batch_size if k == "train" else eval_batch_size
            loaders[k] = DataLoader(dataset, batch_size, shuffle=True,
                                    drop_last=True)

    else:
        raise ValueError("Invalid token_type.")

    return loaders

def load_pretrained_model(name):
    model_path = os.path.join(TRAINED_PATH, "{}.model".format(name))
    model_conf_path = os.path.join(TRAINED_PATH, "{}.conf".format(name))
    model = torch.load(model_path)
    model_conf = pickle.load(open(model_conf_path, 'rb'))

    return model, model_conf


# pre_trained가져오기
def get_pretrained(pretrained):
    if isinstance(pretrained, list):
        pretrained_models = []
        pretrained_config = None
        for pt in pretrained:
            pretrained_model, pretrained_config = load_pretrained_model(pt)
            pretrained_models.append(pretrained_model)
        return pretrained_models, pretrained_config
    else:
        pretrained_model, pretrained_config = load_pretrained_model(pretrained)
        return pretrained_model, pretrained_config

# parse 파일 열기
def parse_e_c(data_file):
    """

    Returns:
        X: a list of tweets
        y: a list of lists corresponding to the emotion labels of the tweets

    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
    X = [d[1] for d in data]
    # dict.values() does not guarantee the order of the elements
    # so we should avoid using a dict for the labels
    y = [[int(l) for l in d[2:]] for d in data]

    return X, y

# dataset parse하기하기
def parse(task, dataset, emotion=None):
    
    E_C = {
        'train': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-train.txt'),
        'dev': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-dev.txt'),
        'gold': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-test-gold.txt')
    }

    if task == 'E-c':
        data_train = E_C[dataset]
        X, y = parse_e_c(data_train)
        return X, y
    else:
        return None, None

# 캐시에 쓰기
def write_cache_word_vectors(file, data):
    with open(file_cache_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

# Embedding 만들기
def load_word_vectors(file, dim):
    if os.path.exists(file):
        print('Indexing file {} ...'.format(file))

        word2idx = {}  # dictionary of words to ids
        idx2word = {}  # dictionary of ids to words
        embeddings = []  # the word embeddings matrix
        embeddings.append(numpy.zeros(dim))
        # flag indicating whether the first row of the embeddings file
        # has a header
        header = False

        # read file, line by line
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):

                # skip the first row if it is a header
                if i == 1:
                    if len(line.split()) < dim:
                        header = True
                        continue

                values = line.split(" ")
                word = values[0]
                vector = numpy.asarray(values[1:], dtype='float32')

                index = i - 1 if header else i

                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)

            # add an unk token, for OOV words
            if "<unk>" not in word2idx:
                idx2word[len(idx2word) + 1] = "<unk>"
                word2idx["<unk>"] = len(word2idx) + 1
                embeddings.append(
                    numpy.random.uniform(low=-0.05, high=0.05, size=dim))

            print(set([len(x) for x in embeddings]))

            print('Found %s word vectors.' % len(embeddings))
            embeddings = numpy.array(embeddings, dtype='float32')

        # write the data to a cache file
        write_cache_word_vectors(file, (word2idx, idx2word, embeddings))

        return word2idx, idx2word, embeddings

    else:
        print("{} not found!".format(file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), file)

#  임베딩 가져오기
def load_embeddings(model_conf):
    word_vectors = os.path.join(BASE_PATH, "embeddings",
                                "{}.txt".format(model_conf["embeddings_file"]))
    word_vectors_size = model_conf["embed_dim"]

    # load word embeddings
    print("loading word embeddings...")
    return load_word_vectors(word_vectors, word_vectors_size)

# definition of trainer
def define_trainer(task,
                   config,
                   name,
                   datasets,
                   monitor,
                   ordinal=False,
                   pretrained=None,
                   finetune=None,
                   label_transformer=None,
                   disable_cache=False):
    """

    Args:
        task (): available tasks
                - "clf": multiclass classification
                - "bclf": binary classification
                - "mclf": multilabel classification
                - "reg": regression
        config ():
        name ():
        datasets ():
        monitor ():
        ordinal ():
        pretrained ():
        finetune ():
        label_transformer ():
        disable_cache ():

    Returns:

    """
    ########################################################################
    # Load pre:trained models
    ########################################################################

    if task == "bclf":
        task = "clf"

    pretrained_models = None
    pretrained_config = None
    if pretrained is not None:
        pretrained_models, pretrained_config = get_pretrained(pretrained)

    if pretrained_config is not None:
        _config = pretrained_config
    else:
        _config = config

    ########################################################################
    # Load embeddings
    ########################################################################
    word2idx = None
    if _config["token_type"] == "word":
        word2idx, idx2word, embeddings = load_embeddings(_config)

    ########################################################################
    # DATASET
    # construct the pytorch Datasets and Dataloaders
    ########################################################################
    loaders = load_datasets(datasets,
                            train_batch_size=_config["batch_train"],
                            eval_batch_size=_config["batch_eval"],
                            token_type=_config["token_type"],
                            params=None if disable_cache else name,
                            word2idx=word2idx,
                            label_transformer=label_transformer)

    ########################################################################
    # MODEL
    # Define the model that will be trained and its parameters
    ########################################################################
    out_size = 1
    if task == "clf":
        classes = len(set(loaders["train"].dataset.labels))
        out_size = 1 if classes == 2 else classes
    elif task == "mclf":
        out_size = len(loaders["train"].dataset.labels[0])

    num_embeddings = None

    if _config["token_type"] == "char":
        num_embeddings = len(loaders["train"].dataset.char2idx) + 1
        embeddings = None

    model = ModelWrapper(embeddings=embeddings,
                         out_size=out_size,
                         num_embeddings=num_embeddings,
                         pretrained=pretrained_models,
                         finetune=finetune,
                         **_config)
    model.to(DEVICE)
    print(model)

    if task == "clf":
        weights = class_weigths(loaders["train"].dataset.labels,
                                to_pytorch=True)
    if task == "clf":
        weights = weights.to(DEVICE)

    ########################################################################
    # Loss function and optimizer
    ########################################################################
    if task == "clf":
        if out_size > 2:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "reg":
        criterion = torch.nn.MSELoss()
    elif task == "mclf":
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        raise ValueError("Invalid task!")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 weight_decay=config["weight_decay"])

    ########################################################################
    # Trainer
    ########################################################################
    if task == "clf":
        pipeline = get_pipeline("bclf" if out_size == 1 else "clf", criterion)
    else:
        pipeline = get_pipeline("reg", criterion)

    metrics, monitor_metric, mode = get_metrics(task, ordinal)

    checkpoint = Checkpoint(name=name, model=model, model_conf=config,
                            monitor=monitor, keep_best=True, scorestamp=True,
                            metric=monitor_metric, mode=mode,
                            base=config["base"])
    early_stopping = EarlyStop(metric=monitor_metric, mode=mode,
                               monitor=monitor,
                               patience=config["patience"])

    trainer = Trainer(model=model,
                      loaders=loaders,
                      task=task,
                      config=config,
                      optimizer=optimizer,
                      pipeline=pipeline,
                      metrics=metrics,
                      use_exp=True,
                      inspect_weights=False,
                      checkpoint=checkpoint,
                      early_stopping=early_stopping)

    return trainer


# trainning

def model_training(trainer, epochs, unfreeze=0, checkpoint=False):
    print("Training...")
    for epoch in range(epochs):
        trainer.train()
        trainer.eval()

        if unfreeze > 0:
            if epoch == unfreeze:
                print("Unfreeze transfer-learning model...")
                subnetwork = trainer.model.feature_extractor
                if isinstance(subnetwork, ModuleList):
                    for fe in subnetwork:
                        unfreeze_module(fe.encoder, trainer.optimizer)
                        unfreeze_module(fe.attention, trainer.optimizer)
                else:
                    unfreeze_module(subnetwork.encoder, trainer.optimizer)
                    unfreeze_module(subnetwork.attention, trainer.optimizer)

        print()

        if checkpoint:
            trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break

# trainning 기록하기
def log_training(self, name, desc):

        results = {}
        scores = {k: v for k, v in self.scores.items() if k != "loss"}

        results["name"] = name
        results["desc"] = desc
        results["scores"] = scores

        path = os.path.join(EXPS_PATH, self.config["name"])

        ####################################
        # JSON
        ####################################
        json_file = path + ".json"
        try:
            with open(json_file) as f:
                data = json.load(f)
        except:
            data = []

        data.append(results)

        with open(json_file, 'w') as f:
            json.dump(data, f)

        ####################################
        # CSV
        ####################################
        _results = []
        for result in data:
            _result = {k: v for k, v in result.items() if k != "scores"}
            for score_name, score in result["scores"].items():
                for tag, values in score.items():
                    _result["_".join([score_name, tag, "min"])] = min(values)
                    _result["_".join([score_name, tag, "max"])] = max(values)
            _results.append(_result)

        with open(path + ".csv", 'w') as f:
            pandas.DataFrame(_results).to_csv(f, sep=',', encoding='utf-8')



#################################[3]프로세스 실행

model_config = TASK1_EC
    task = 'E-c'
    # load the dataset and split it in train and val sets
    X_train, y_train = parse(task=task, dataset="train")
    X_dev, y_dev = parse(task=task, dataset="dev")
    X_test, y_test = parse(task=task, dataset="gold")

    datasets = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "gold": (X_test, y_test),
    }


# trainer 설정하기
name = model_config["name"]
trainer = define_trainer("mclf", config=model_config, name=name,
                         datasets=datasets,
                         monitor="dev",
                         pretrained=pretrained,
                         finetune=finetune)

# trainning 하기
model_training(trainer, model_config["epochs"], unfreeze=unfreeze)


