# 내장
import logging

# 서드파티
import pandas as pd
import matplotlib.pyplot as plt
from ludwig.api import LudwigModel
from ludwig.visualize import learning_curves
from ludwig.visualize import confusion_matrix
from ludwig.contribs.mlflow import MlflowCallback

# 프로젝트
from src.model.ludwig.config import LudwigConfigManager
from src.preprocessing.sms import SMSDataPreprocessingManager


def main():
    prep_manager = SMSDataPreprocessingManager(
        feature_column_name='message',
        label_column_name='label'
    )
    path = './data/sample/spam.csv'
    df = prep_manager.read_sample_data(path, 1.0)
    df = prep_manager.split(df, 0.8)
    train_df, test_df = df

    # Constructs Ludwig model from config dictionary
    config_manager = LudwigConfigManager(prep_manager)
    model = LudwigModel(config_manager.get_ludwig_config(),
                        logging_level=logging.INFO,
                        callbacks=[MlflowCallback()])

    # Trains the model. This cell might take a few minutes.
    train_stats, preprocessed_data, output_directory = model.train(
        dataset=train_df,
        experiment_name=config_manager.experiment_name,
        model_name=config_manager.model_name,
        output_directory=config_manager.saved_model_base_dir,
    )

    # Generates predictions and
    # performance statistics for the test set.
    test_stats, predictions, output_directory = model.evaluate(
        dataset=test_df,
        collect_predictions=True,
        collect_overall_stats=True,
        output_directory=config_manager.model_info_output_dir,
    )

    # Visualizes learning curves, which show how
    # performance metrics changed over time
    # during training.
    learning_curves(
        train_stats,
        output_feature_name='class',
        output_directory=config_manager.model_info_output_dir,
        file_format='png'
    )

    confusion_matrix(
        [test_stats],
        model.training_set_metadata,
        output_feature_name='class',
        top_n_classes=[5],
        model_names=[''],
        normalize=True,
        output_directory=config_manager.model_info_output_dir,
        file_format='png',
    )

    text_to_predict = pd.DataFrame({
        f"{prep_manager.feature_column_name}": [
            "Google may spur cloud cybersecurity M&A with $5.4B Mandiant buy",
            "Europe struggles to meet mounting needs of Ukraine's fleeing millions",
            "How the pandemic housing market spurred buyer's remorse across America",
        ]
    })
    predictions, output_directory = model.predict(text_to_predict)
    print(predictions)

    return predictions


if __name__ == '__main__':
    main()
