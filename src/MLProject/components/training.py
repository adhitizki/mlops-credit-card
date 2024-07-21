from MLProject import logger
from MLProject.entity.config_entity import TrainingConfig

import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def objective(self, trial):
        # Define hyperparameters to optimize
        C = trial.suggest_float('C', *self.config.params_C, log=True)
        solver = trial.suggest_categorical('solver', self.config.params_solver)
        
        # Initialize and train the Logistic Regression classifier
        model = LogisticRegression(
            C=C,
            solver=solver,
            random_state=42
        )
        
        # Fit the model on the training data
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predict on the validation set
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Compute the F1 score for class 1
        f1 = f1_score(self.y_test, y_test_pred, labels=[1], average='binary')
        return f1

    def hpo_logistic_regression(self) -> None:
        """train the data with random forest model using hyperparameter optimization and dump the data
        """
        logger.info(f"Load scaled data train from {self.config.scaled_train_path}.")
        self.X_train_scaled = joblib.load(self.config.scaled_train_path)

        logger.info(f"Load scaled data test from {self.config.scaled_test_path}.")
        self.X_test_scaled = joblib.load(self.config.scaled_test_path)
        
        logger.info(f"Load data train output from {self.config.output_train_path}.")
        self.y_train = joblib.load(self.config.output_train_path)

        logger.info(f"Load data test output from {self.config.output_test_path}.")
        self.y_test = joblib.load(self.config.output_test_path)

        logger.info(f"Find best parameter using hyperparameter optimization")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.config.params_n_trials)

        logger.info(f"Get best parameter")
        best_params = study.best_params
        
        logger.info(f"Train the model.")
        model = LogisticRegression(
            C=best_params['C'],
            solver=best_params['solver'],
            random_state=42
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        logger.info(f"Dump the model.")
        joblib.dump(model, self.config.model_path)