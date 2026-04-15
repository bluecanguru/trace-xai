datasets = ["heart", "diabetes", "stroke", "hba1c", "ckd"]

for dataset in datasets:
    data = load_dataset(dataset)
    target = get_target(dataset)

    run_experiment(dataset, data, target)