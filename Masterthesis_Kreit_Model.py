import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import warnings
from sklearn.exceptions import DataConversionWarning
import math
import logging
import pandas as pd
import pickle
import json
from datetime import datetime
from datetime import date

import sys
from numpy import median, nanmedian
!{sys.executable} -m pip install shap
!{sys.executable} -m pip install catboost

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import shap

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.display.max_columns = 200
pd.set_option('mode.chained_assignment', None)


def preprocess_training_data(df_train_data, target_date):
    """ Preprocess training data
    """
    # Map columns to VDA Client format
    mapping_lot = {
                'initialRegistrationYear': 'initial_registration_year',
                'initialRegistrationMonth': 'initial_registration_month',
                'odometer': 'odometer',
                'odometerUnit': 'odometer_unit',
                'currencyUnit': 'currency_unit',
                'vin': 'vin',
                'brand': 'make',
                'model': 'model',
                'vehicleSegment': 'vehicle_segment',
                'bodyStyleSalesName': 'body_style_brand',
                'bodyStyle': 'body_style',
                'trim': 'trim',
                'firstDate': 'first_date',
                'firstPrice': 'first_price',
                'lastDate': 'last_date',
                'lastPrice': 'last_price',
                'engineType': 'engine_base_type',
                'version': 'version',
                'engineTechnology': 'engine_type',
                'engineSize': 'engine_size_value',
                'enginePower': 'engine_power_value',
                'engineCylinderCount': 'engine_cylinders',
                'engineEfficiencySalesName': 'engine_efficiency',
                'fuelType': 'fuel_type',
                'drivetrainSalesName': 'drivetrain_brand',
                'drivetrain': 'drivetrain',
                'gearCount': 'num_gears',
                'transmissionSalesName': 'transmission_brand',
                'modelVariant': 'model_variant',
                'transmission': 'transmission',
                'bodyColorName': 'body_color_name',
                'bodyColorFinish': 'body_color_finish',
                'interiorColorSeats': 'interior_color_seats',
                'towBar': 'tow_bar',
                'airSuspension': 'air_suspension',
                'airConditioning': 'air_conditioning',
                'emergencyAssist': 'emergency_assist',
                'highBeamAssist': 'high_beam_assist',
                'fatigueDetection': 'fatigue_detection',
                'parkingAssist': 'parking_assist',
                'laneKeepingAssist': 'lane_keeping_assist',
                'laneChangeAssist': 'lane_change_assist',
                'blindSpotAssist': 'blind_spot_assist',
                'trailerAssist': 'trailer_assist',
                'frontAssist': 'front_assist',
                'cruiseControl': 'cruise_control',
                'parkDistanceControl': 'park_distance_control',
                'rearCam': 'rear_cam',
                'centerConsoleArmrest': 'armrest_console_in_center',
                'mirrorLink': 'mirror_link',
                'navigationSystem': 'navigation_system',
                'soundSystem': 'sound_system',
                'headlights': 'headlights',
                'sunroof': 'sunroof',
                'telephone': 'telephone',
                'doorCount': 'door_count',
                'seatCount': 'seat_count',
                'seatHeating': 'seat_heating',
                'seatCover': 'leather_partial_leather',
                'seatVentilation': 'seat_ventilation',
                'seatMassageFunction': 'seat_massage_function',
                'sportSeats': 'sport_seats',
                'parkingHeater': 'parking_heater',
                'extendedWarranty': 'extended_warranty',
                'euroEmissionsStandard': 'euro_emissions_standard',
                'registrationAsTruck': 'registration_as_truck',
                'steeringWheelPlacement': 'steering_wheel_placement',
                'runFlatTires': 'run_flat_tires',
                'fodTrafficSignRecognition': 'fod_traffic_sign_recognition',
                'fodNavigationSystem': 'fod_navigation_system',
                'fodParkingAssist': 'fod_parking_assist',
                'fodLaneChangeAssist': 'fod_lane_change_assist',
                'fodServotronicSteering': 'fod_servotronic_steering',
                'fodDigitalRadioReceiverDab': 'fod_digital_radio_receiver_dab',
                'fodKeylessEntry': 'fod_keyless_entry',
                'fodLightAssistanceSystems': 'fod_light_assistance_systems',
                'fodAutomaticCruiseControl': 'fod_automatic_cruise_control'
             }
    
    # map_lot_columns(df_train_data, mapping_lot)
    print("Renaming columns...")
    df_train_data = df_train_data.rename(columns=mapping_lot, errors="raise")
    
    # Filter data
    print("Filtering data...")
    df_train_data = filter_data(df_train_data, target_date)

    return df_train_data

def filter_data(df_in, price):    
    today = date.today()
    df_out = df_in.copy() 
    
    df_out = df_out[df_out.initial_registration_year.notna()]
    df_out = df_out[df_out.initial_registration_month.notna()]
    df_out = df_out[df_out.engine_power_value.notna()]
    df_out = df_out[df_out.engine_size_value.notna()]
    df_out = df_out[df_out.door_count.notna()]
    df_out = df_out[df_out.seat_count.notna()]
    if price == "last_price":
        df_out = df_out[df_out.last_price.notna()]
    elif price == "first_price":
        df_out = df_out[df_out.first_price.notna()]
    
    df_out = df_out[df_out.usageState == 'used']
    df_out = df_out[(df_out.make == 'Volkswagen') | (df_out.make == 'Audi') | (df_out.make == 'Seat') | (df_out.make == 'Skoda') | (df_out.make == 'Volkswagen Commercial Vehicles')]
    df_out.last_date = df_out.last_date.astype('datetime64[ns]')
    df_out = df_out[(df_out.initial_registration_year >= (df_out.last_date.dt.year - 8)) & (df_out.initial_registration_year <= df_out.last_date.dt.year)]
    df_out = df_out[df_out.odometer >= 0]
    if price == "last_price":
        df_out = df_out[(df_out.last_price >= 0) & (df_out.last_price <= 500000)]
    elif price == "first_price":
        df_out = df_out[(df_out.first_price >= 0) & (df_out.first_price <= 500000)]
    df_out = df_out[(df_out.door_count >= 1) & (df_out.door_count <= 10)]
    df_out = df_out[df_out.seat_count > 1]
    df_out = df_out[(df_out.engine_size_value >= 0.25) & (df_out.engine_size_value <= 10)]
    df_out = df_out[(df_out.engine_power_value > 0) & ((df_out.engine_power_value <= 400) | ((df_out.engine_power_value >= 400) & (df_out.engine_size_value >= 3)) )]
    df_out = df_out[df_out.last_date >= np.datetime64(str(today.year)) - np.timedelta64(2, 'Y')]
    
    entries_removed = df_in.shape[0] - df_out.shape[0]
    print(f"{entries_removed:,}", "entries removed")
    print("Count output entries: " + str(df_out.shape[0]))
    
    return df_out

def feature_engineering(df, date_feature):
    """
    Create new features.
    """
    
    df_out = df.copy()

    today = date.today()
    df_out.loc[:, 'initial_registration_date'] = df_out['initial_registration_year'].astype("int32").astype(str) +"-"+ df_out["initial_registration_month"].astype("int32").astype(str)+'-01'
    df_out.loc[:, 'age'] = pd.to_datetime(df_out.loc[:, date_feature]) - pd.to_datetime(df_out.initial_registration_date)
    df_out['age'] = df_out['age'].apply(lambda x: x.days)
    df_out.loc[:, 'initial_registration_timedelta'] = (today.year - df_out.initial_registration_year) * 12 + today.month - df_out.initial_registration_month
    df_out.loc[:, 'initial_registration_month_sin'] = np.sin(df_out.initial_registration_month.astype("float") / 12 * 2 * math.pi)
    df_out.loc[:, 'initial_registration_month_cos'] = np.cos(df_out.initial_registration_month.astype("float") / 12 * 2 * math.pi)
    df_out.loc[:, 'sales_year'] = pd.to_datetime(df_out.loc[:, date_feature]).dt.year
    df_out.loc[:, 'sales_month_sin'] = np.sin(pd.to_datetime(df_out.loc[:, date_feature]).dt.month.astype("float") / 12 * 2 * math.pi)
    df_out.loc[:, 'sales_month_cos'] = np.cos(pd.to_datetime(df_out.loc[:, date_feature]).dt.month.astype("float") / 12 * 2 * math.pi)
    df_out.loc[:, 'sales_date_timedelta'] = (today.year - pd.to_datetime(df_out.loc[:, date_feature]).dt.year) * 12 + today.month - pd.to_datetime(df_out.loc[:, date_feature]).dt.month
    
    # fill missing values in categorical features with "unknown"
    print("Filling missing categorical features...")
    df_out = fill_missing_categorical_features(df_out)

    return df_out

def fill_missing_categorical_features(df):
    """
    Fill missing values in categorical features with "unknown".
    """

    cat_features = [
        'make',
        'model',
        'vehicle_segment',
        'body_style_brand',
        'body_style',
        'trim',
        'engine_base_type',
        'engine_type',
        'engine_efficiency',
        'fuel_type',
        'drivetrain_brand',
        'drivetrain',
        'transmission_brand',
        'transmission',
        'body_color_name',
        'body_color_finish',
        'interior_color_seats',
        'tow_bar',
        'air_conditioning',
        'headlights',
        'euro_emissions_standard',
        'navigation_system',
        'park_distance_control',
        'sport_seats',
        'model_variant'
    ]

    df[cat_features] = df[cat_features].fillna('unknown')

    return df

def feature_selection(df_train_data, target):
    """
    Cast and select features for training.
    """

    # Data type casting
    print("Data type casting...")
    df_train_data = data_type_casting(df_train_data, target=target)

    # Select features in correct order for model
    keep_columns = [
        'odometer',
        'make',
        'model',
        'vehicle_segment',
        'body_style_brand',
        'body_style',
        'trim',
        'engine_base_type',
        'engine_type',
        'engine_power_value',
        'engine_efficiency',
        'fuel_type',
        'engine_size_value',
        'drivetrain_brand',
        'drivetrain',
        'transmission_brand',
        'transmission',
        'body_color_name',
        'body_color_finish',
        'interior_color_seats',
        'tow_bar',
        'air_conditioning',
        'park_distance_control',
        'headlights',
        'door_count',
        'seat_count',
        'euro_emissions_standard',
        'age',
        'navigation_system',
        'emergency_assist',
        'high_beam_assist',
        'fatigue_detection',
        'parking_assist',
        'lane_keeping_assist',
        'lane_change_assist',
        'blind_spot_assist',
        'seat_heating',
        'model_variant',
        'sunroof',
        'leather_partial_leather',
        'parking_heater',
        'sport_seats',
        'sales_month_sin',
        'sales_month_cos',
        'sales_date_timedelta',
        'initial_registration_month_sin',
        'initial_registration_month_cos',
        'initial_registration_timedelta'
    ]
    
    # For regression model first_price
    if target == "first_price":
        keep_columns = keep_columns + ['first_date', 'first_price']
    # For regression model last_price
    elif target == "last_price":
        keep_columns = keep_columns + ['last_date', 'last_price']
        
    return df_train_data[keep_columns]

def data_type_casting(df, target):
    """
    Because we read from csv, we need to cast the columns to the correct data types.
    """

    dtypes = {
        'initial_registration_timedelta': int,
        'initial_registration_month_sin': float,
        'initial_registration_month_cos': float,
        'odometer': float,
        'make': str,
        'model': str,
        'vehicle_segment': str,
        'body_style_brand': str,
        'body_style': str,
        'trim': str,
        'engine_base_type': str,
        'engine_type': str,
        'engine_power_value': float,
        'engine_efficiency': str,
        'fuel_type': str,
        'engine_size_value': float,
        'drivetrain_brand': str,
        'drivetrain': str,
        'transmission_brand': str,
        'transmission': str,
        'body_color_name': str,
        'body_color_finish': str,
        'interior_color_seats': str,
        'tow_bar': str,
        'air_conditioning': str,
        'headlights': str,
        'door_count': float,
        'seat_count': float,
        'euro_emissions_standard': str,
        'sales_month_sin': float,
        'sales_month_cos': float,
        'sales_date_timedelta': int,
        'age': int,
        'navigation_system': str,
        'emergency_assist': bool,
        'high_beam_assist': bool,
        'fatigue_detection': bool,
        'parking_assist': bool,
        'lane_keeping_assist': bool,
        'lane_change_assist': bool,
        'blind_spot_assist': bool,
        'seat_heating': bool,
        'park_distance_control': str,
        'parking_heater': bool,
        'sport_seats': str,
        'model_variant': str,
        'sunroof': bool,
        'leather_partial_leather': bool,
        'first_date': 'datetime64[ns]',
        'first_price': float,
    }
    
    # For regression model first_price
    if target == "first_price":
        dtypes.update({'first_date': 'datetime64[ns]', 'first_price': float})
    # For regression model last_price
    elif target == "last_price":
        dtypes.update({'last_date': 'datetime64[ns]', 'last_price': float})

    for col in dtypes.keys():
        df[col] = df[col].astype(dtypes[col])

    return df

def train_test_split(df_train, model_settings, target, date, drop_more_columns=None):
    """ Perform a train/test split or use the full data for the model training.
    """
    train_mode = model_settings['train_mode']
    test_weeks = 4
    min_date = df_train[date].min()
    max_date = df_train[date].max()
    
    drop_columns = [target, date]
    if drop_more_columns is not None:
        drop_columns = drop_columns + drop_more_columns
        
    print(drop_columns)

    print(f"First date in training data set: {min_date}")
    print(f"Latest date in training data set: {max_date}")

    if train_mode == "eval":

        print(f"Performing a train/test split with last {test_weeks} weeks.")
        train_end = max_date - pd.Timedelta(weeks=test_weeks)

        # Train on 23 consecutive months, test on last 4 weeks
        train_period = df_train[df_train[date] <= train_end]
        test_period = df_train[df_train[date] > train_end]

        x_train, y_train = train_period.drop(drop_columns, axis=1), train_period[target]
        x_test, y_test = test_period.drop(drop_columns, axis=1), test_period[target]

        print(f"Using {len(x_train)} training data from {min_date.strftime('%Y-%m-%d')} until {train_end.strftime('%Y-%m-%d')} and {len(x_test)} testing data.")

    elif train_mode == "prod":

        # Train on 24 consecutive months
        train_end = max_date - pd.Timedelta(days=1)
        train_period = df_train[df_train[date] <= train_end]

        x_train, y_train = train_period.drop(drop_columns, axis=1), train_period[target]
        x_test, y_test = None, None

        print(f"Using full {len(x_train)} training data from {min_date.strftime('%Y-%m-%d')} until {train_end.strftime('%Y-%m-%d')}.")
    else:  # pragma: no cover
        raise NotImplementedError(f"Wrong parameter for train_mode {train_mode}")

    print(f"Model features: {x_train.columns}")

    return x_train, y_train, x_test, y_test 

def create_model(model_settings):
    """
    Create the Catboost model.
    """
    
    model_type = model_settings['model_type']
    loss = model_settings['loss']
    assert loss in ["standard", "uncertainty"], f"Loss function {loss} is not implemented."

    # which loss function to use
    if loss == "standard":
        if model_type == "regression":
            cb_model = CatBoostRegressor(task_type="CPU",
                                     n_estimators=model_settings['iterations'],
                                     one_hot_max_size=model_settings['one_hot_max_size'],
                                     max_depth=model_settings['max_depth'],
                                     learning_rate=model_settings['learning_rate'],
                                     loss_function='RMSE',
                                     train_dir=model_settings['checkpoint_path']
                                     )
    elif loss == "custom":  # pragma: no cover
        raise NotImplementedError(f"Loss function {loss} is not implemented.")
    elif loss == "quantile":  # pragma: no cover
        raise NotImplementedError(f"Loss function {loss} is not implemented.")
    elif loss == "uncertainty":
        if model_type == "regression":
            cb_model = CatBoostRegressor(task_type="CPU",
                                     n_estimators=model_settings['iterations'],
                                     one_hot_max_size=model_settings['one_hot_max_size'],
                                     max_depth=model_settings['max_depth'],
                                     learning_rate=model_settings['learning_rate'],
                                     loss_function='RMSEWithUncertainty',
                                     posterior_sampling=True,
                                     train_dir=model_settings['checkpoint_path']
                                     )

    else:  # pragma: no cover
        raise NotImplementedError("Model object could not be created.")

    print("Catboost model object is created.")

    return cb_model

def train_model(model, model_settings, x_train, y_train):
    """
    Perform the model fitting.
    """

    weight_arr = calculate_weights(model_settings, len(x_train))
    cat_features = np.where(x_train.dtypes == object)[0]

    # Enable checkpointing with defined number of seconds to recover from interruptions during spot training
    use_checkpoints = model_settings['checkpoint_interval'] > 0

    print("Start training...")
    if weight_arr:
        model.fit(x_train, y_train, cat_features=cat_features, verbose=True, sample_weight=weight_arr,
                 #      save_snapshot=use_checkpoints, snapshot_file=model_settings['checkpoint_path'] + "checkpoint",
                 #      snapshot_interval=model_settings['checkpoint_interval']
                 )
    else:
        model.fit(x_train, y_train, cat_features=cat_features, verbose=True,
                 #      save_snapshot=use_checkpoints, snapshot_file=model_settings['checkpoint_path'] + "checkpoint",
                 #      snapshot_interval=model_settings['checkpoint_interval']
                 )

    print("Training finished.")

def calculate_weights(model_settings, length_train_data):
    """
    Calculate weights for exponential weight decay.
    """

    # Exponential decay
    if model_settings['weight_decay']:
        print(f"Calculating {length_train_data} weights with parameter weight_decay={model_settings['weight_decay']}")

        k = (-1) * model_settings['weight_decay']  # -8*(10.0)**(-6)

        weight_arr = []
        for i in range(length_train_data):
            weight_arr.append(100 ** math.exp(k * i))
        weight_arr = list(reversed(weight_arr))
    else:
        weight_arr = None

    return weight_arr

def calc_mape(y_actual, y_predicted):
    """
    Calculate the mean absolute percentage error (MAPE).
    """

    assert len(y_actual) == len(y_predicted), "Length mismatch!"
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100

    return mape 

def evaluate_model(model, x_test, y_test, mode, target):
    """
    Calculate the relevant model performance metrics and print it to the logs.
    """

    prediction = model.predict(x_test)

    if model.get_params().get("loss_function", "") == "RMSEWithUncertainty":
        x_test["prediction"] = prediction[:, 0]   # first return value is prediction
        x_test["uncertainty"] = prediction[:, 1]  # second return value is variance / uncertainty
        print(f"mean_standard_deviation={np.round(np.mean(x_test['uncertainty'].apply(lambda x: math.sqrt(x))),4)};")
        print(f"median_standard_deviation={np.round(np.median(x_test['uncertainty'].apply(lambda x: math.sqrt(x))), 4)};")
    else:
        x_test["prediction"] = prediction

    # save data including predictions to output
    x_test[target] = y_test
    x_test.to_pickle(model_settings['output_path'] + f"data_{mode}_with_predictions.pkl")

    y_pred = x_test["prediction"]

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calc_mape(y_test, y_pred)
    mean_error = np.mean(y_pred - y_test)
    median_error = np.median(y_pred - y_test)

    metrics = {
        f"r2_{mode}": r2,
        f"mae_{mode}": mae,
        f"mape_{mode}": mape,
        f"mean_error_{mode}": mean_error,
        f"median_error_{mode}": median_error
    }

    print(f"Model evaluation metrics for {mode} mode:")
    for key in metrics.keys():
        print(f"{key}={np.round(metrics[key],4)};")

    return metrics

def save_model(model_settings, model, target):
    # Save model as pickle file
    today = date.today()
    
    output_filepath = model_settings['output_path'] + "model" + str(today.year) +"_" + str(today.month) + "_" + str(today.day) + "_" + target + ".pkl"
    pickle.dump(model, open(output_filepath, 'wb'))

    print(f"Model file is saved to {output_filepath}")

def save_evaluation(model_settings, metrics_result, target):
    # write metrics file to output folder
    today = date.today()
    with open(model_settings['output_path'] + 'metrics' + str(today.year) +"_" + str(today.month) + "_" + str(today.day) + "_" + target + '.json5', 'w') as fp:
        json.dump(metrics_result, fp)


# Train first_price model
df_cars = preprocess_training_data(df_cars, "first_price")
df_cars = feature_engineering(df_cars, "first_date")
df_cars = feature_selection(df_cars, target="first_price")
df_cars.to_pickle("output/df_cars_first_price.pkl")

# Set model settings
current_model_settings = {
                    'model_type': 'regression',
                    'loss': 'uncertainty',
                    'iterations': 2000,
                    'one_hot_max_size': 20,
                    'max_depth': 15,
                    'learning_rate': 0.01,
                    'weight_decay': 8e-06,
                    'train_mode': 'eval',   #Controlled, if test set is generated
                    'checkpoint_interval': 900,
                    'checkpoint_path': "checkpoints/",
                    'output_path': "output/"
                 }

x_train, y_train, x_test, y_test = train_test_split(df_cars, current_model_settings, "first_price", "first_date")
model_first_price = create_model(current_model_settings)
train_model(model_first_price, current_model_settings, x_train, y_train)
save_model(current_model_settings, model_first_price, target="first_price")

# Evaluation for training and test sets
metrics = evaluate_model(model_first_price, x_train, y_train, mode="train", target="first_price")
metrics_result = {"train_set": metrics}
if x_test is not None:
    # out-of-sample metrics if not full data are used for training
    metrics = evaluate_model(model_first_price, x_test, y_test, mode="test", target="first_price")
    metrics_result["test_set"] = metrics
save_evaluation(current_model_settings, metrics_result, target="first_price")

# Train first_price model
df_cars = preprocess_training_data(df_cars, "last_price")
df_cars = feature_engineering(df_cars, "last_date")
df_cars = feature_selection(df_cars, target="last_price")

# Add predicted first_price as feature
df_first_price_predict = df_cars.drop(["last_price", "last_date"], axis=1)
first_price_predict = model_first_price.predict(df_first_price_predict)
df_cars['first_price_predict'] = first_price_predict[:, 0]
df_cars['first_price_predict'] = df_cars['first_price_predict'].astype(int).astype(float)
df_cars.to_pickle("output/df_cars_last_price.pkl")

x_train, y_train, x_test, y_test = train_test_split(df_cars, current_model_settings, "last_price", "last_date")
model_last_price = create_model(current_model_settings)
train_model(model_last_price, current_model_settings, x_train, y_train)
save_model(current_model_settings, model_last_price, target="last_price")

# Evaluation for training and test sets
metrics = evaluate_model(model_last_price, x_train, y_train, mode="train", target="last_price")
metrics_result = {"train_set": metrics}
if x_test is not None:
    # out-of-sample metrics if not full data are used for training
    metrics = evaluate_model(model_last_price, x_test, y_test, mode="test", target="last_price")
    metrics_result["test_set"] = metrics

save_evaluation(current_model_settings, metrics_result, target="last_price")

# Print error level
df_test_w_preds = x_test
df_test_w_preds['last_price'] = y_test
df_test_w_preds['prediction'] = model_last_price.predict(df_test_w_preds)[:, 0]
df_test_w_preds['prediction'] = df_test_w_preds['prediction'].astype(int).astype(float)
df_test_w_preds['error_level'] = np.round((df_test_w_preds.prediction - df_test_w_preds.last_price) / df_test_w_preds.last_price, 5)
df_test_w_preds['error_level_abs'] = np.round(np.abs(df_test_w_preds.prediction - df_test_w_preds.last_price) / df_test_w_preds.last_price, 5)
count = df_test_w_preds[df_test_w_preds.error_level_abs > 0].shape[0]

df_error_level_abs = df_test_w_preds[df_test_w_preds.error_level_abs > 0].groupby('error_level_abs')['error_level_abs'].count()
df_error_level_under = df_test_w_preds[df_test_w_preds.error_level < 0].groupby('error_level')['error_level'].count()
df_error_level_over = df_test_w_preds[df_test_w_preds.error_level > 0].groupby('error_level')['error_level'].count()

df_error_level_abs_pct = df_error_level_abs / count
df_error_level_under_pct = df_error_level_under / count
df_error_level_over_pct = df_error_level_over / count

# Print plot picture 5-4
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_error_level_abs_pct.index, df_error_level_abs_pct, color='blue', label='Total absolute')
ax.plot(df_error_level_under_pct.index * (-1), df_error_level_under_pct, color='green', label='Undershooting')
ax.plot(df_error_level_over_pct.index, df_error_level_over_pct, color='red', label='Overshooting')
ax.legend()

plt.title("Wrong predictions by error level (CatBoost)")
plt.xlabel("error level")
plt.ylabel("perfentage of cars")

plt.xlim([0.025, 0.3])
plt.ylim([0, 0.13])
plt.show()

# Print plot picture 5-5
fig, ax = plt.subplots(figsize=(12, 6))
count_clusters = 16
price_range = 10000

list_all = []
tupel = ()
for i in range(count_clusters):
    min = i*price_range 
    max = (i+1)*price_range
    name_cluster = '<'+ str(int(max/1000))
    if i == count_clusters-1:
        max=99999999
        name_cluster = str(int(min/1000)) + '+'
    
    tupel_clusters = tupel_clusters + (name_cluster,)
    
    median_total = df_test_w_preds[((df_test_w_preds.last_price >= min) & 
                                    (df_test_w_preds.last_price < max) & 
                                   (df_test_w_preds.error_level_abs > 0))].loc[:, 'error_level_abs'].median()
    median_under = df_test_w_preds[((df_test_w_preds.last_price >= min) & 
                                   (df_test_w_preds.last_price < max) &
                                   (df_test_w_preds.error_level < 0))].loc[:, 'error_level'].median() * (-1)
    median_over  = df_test_w_preds[((df_test_w_preds.last_price >= min) &
                                   (df_test_w_preds.last_price < max) &
                                   (df_test_w_preds.error_level > 0))].loc[:, 'error_level'].median()
    list_cluster = [np.round(median_total, 3), np.round(median_under, 3), np.round(median_over, 3)]
    list_all.append(list_cluster)

X = np.arange(count_clusters)
list_all = [list(i) for i in zip(*list_all)]
ax.bar(X - 0.25, list_all[0], color = 'b', width = 0.25)
ax.bar(X       , list_all[1], color = 'g', width = 0.25)
ax.bar(X + 0.25, list_all[2], color = 'r', width = 0.25)    
ax.legend(labels=['Total absolute', 'Undershooting', 'Overshooting'])

plt.title("Error level by clustered predicted_first_price (CatBoost)")
plt.xlabel("predicted_first_price (in Tsd)")
plt.ylabel("median error level")

ax.set_xticks(X, tupel_clusters)
# plt.xlim([0, 100000])
# plt.ylim([0, 1])
plt.show()

# Feature Importances
# Permutation
def plot_feature_importance_permutation(importance, names, model_type):
    
    import seaborn as sns 
  
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    # Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    # Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
  
plot_feature_importance_permutation(model_first_price.get_feature_importance(), x_test.columns,'CATBOOST')

# Shapley Values
df_cars_shapley = pd.concat([x_train, x_test], axis=0)
df_cars_shapley_X = df_cars_shapley
explainer = shap.Explainer(model_last_price)
shap_values = explainer(df_cars_shapley_X)
shap.summary_plot(shap_values, features=df_cars_shapley_X, feature_names = df_cars_shapley_X.columns)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, df_cars_shapley_X)
shap.plots.waterfall(shap_values[0])
shap.initjs()
shap.plots.force(shap_values[0])
shap.plots.bar(shap_values)
