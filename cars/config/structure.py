from pathlib import Path


def get_project_structure():

    root_dir = Path.cwd()
    input_data = root_dir / "cars" / 'input'
    output_data = root_dir / "cars" / 'output'

    return dict(
        input_data=root_dir / input_data,
        output_data=root_dir / output_data,
        logging_dir=output_data / 'logs',
        stanford_data_source=input_data / 'stanford',
    )


def get_data_sources():
    structure = get_project_structure()

    return dict(
        stanford=dict(
            data_source='https://ai.stanford.edu/~jkrause/cars/car_dataset.html',
            train=dict(
                location=structure['stanford_data_source'] / 'cars_train.tgz',
                source='http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
            ),
            test=dict(
                location=structure['stanford_data_source'] / 'cars_test.tgz',
                source='http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
            ),
            labels=dict(
                location=structure['stanford_data_source'] / 'labels' / 'labels_df.csv',
                source='https://raw.githubusercontent.com/morganmcg1/stanford-cars/master/labels_df.csv'
            ),
        ),
    )