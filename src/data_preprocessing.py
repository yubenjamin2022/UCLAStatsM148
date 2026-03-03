import numpy as np
import pandas as pd
from collections import Counter

def flatten_journeys(input_file, output_file):
    """
    Creates a single entry for each unique user journey in the original dataset while preserving all information.
    
    input_file (str): File path to input file containing events.
    output_file (str): The name of the parquet file to be created (e.g. test_data_flattened).
                       Output will go to f'data/{output_file}.parquet'

    """
    print("Flattening journeys\n"
          "-------------------")
    df = pd.read_csv(
        input_file,
        low_memory=True,
        parse_dates=['event_timestamp'],
        usecols=['id', 'ed_id', 'event_name', 'event_timestamp', 'journey_steps_until_end'],
        dtype={
            'ed_id': np.uint8,
            'journey_steps_until_end': np.uint16
        }
    )
    print("Successfully read in file.")

    # Drop any duplicate entries within a journey
    df.drop_duplicates(
        subset=['id', 'ed_id', 'event_timestamp'],
        inplace=True
    )

    # Recompute journey steps until end after dropping duplicates
    df.sort_values(by=['id', 'event_timestamp'], inplace=True)
    df['journey_steps_until_end'] = df.groupby('id').cumcount() + 1
    df.reset_index(drop=True, inplace=True)

    print(f"Dropped duplicate entries and corrected values for journey steps until end.")

    df = (
        df
        .groupby("id", sort=False)
        .agg(
            events=("ed_id", list),
            event_names=("event_name", list),
            timestamps=("event_timestamp", list),
            journey_length=("ed_id", "size")
        )
        .reset_index()
    )

    print("Created a single entry for each journey.")

    df.to_parquet(f"data/{output_file}.parquet", index=False)
    print("Successfully created parquet file.")

def create_journey_features(input_file, output_file):
    """
    Adds features to each journey that will be used to classify journeys as successful or unsuccessful.
    
    input_file (str): File path to input file containing flattened user journeys.
    output_file (str): The name of the parquet file to be created (e.g. test_data_features).
                       Output will go to f'data/{output_file}.parquet'

    """
    print("Creating Features for data\n"
          "--------------------------")
    
    df = pd.read_parquet(input_file)
    df2 = pd.read_parquet("data/train_data2_flattened.parquet")
    print("Successfully read in files.")

    df['customer_id'] = df['id'].str.partition(' ')[0].astype('str')
    df2['customer_id'] = df2['id'].str.partition(' ')[0].astype('str')

    df['start_date'] = pd.to_datetime(df['timestamps'].str[0], utc=True)
    df['end_timestamp'] = pd.to_datetime(df['timestamps'].str[-1], utc=True)
    df2['start_date'] = pd.to_datetime(df2['timestamps'].str[0], utc=True)

    current_datetime = df['end_timestamp'].max()

    df['current_datetime'] = current_datetime
    df['end_timestamp'] = current_datetime


    df2['successful'] = df2['events'].map(lambda x: 28 in x).astype('int8')
    df2 = df2.sort_values(['customer_id', 'start_date'])

    g2 = df2.groupby('customer_id', sort=False)
    df2['cum_successful'] = g2['successful'].cumsum()

    num_previous = np.zeros(len(df), dtype=np.int32)
    num_successful = np.zeros(len(df), dtype=np.int32)

    df_grouped = df.groupby('customer_id')
    df2_grouped = df2.groupby('customer_id')

    for cust_id, df_group in df_grouped:
        if cust_id not in df2_grouped.groups:
            continue

        df2_group = df2_grouped.get_group(cust_id)
        df2_dates = df2_group['start_date'].values
        df_dates = df_group['start_date'].values

        indices = np.searchsorted(df2_dates, df_dates, side='left')
        num_previous[df_group.index] = indices
        successful_cumsum = df2_group['cum_successful'].values

        mask = indices > 0
        num_successful[df_group.index[mask]] = successful_cumsum[indices[mask] - 1]
    
    df['num_previous_journeys'] = num_previous
    df['num_successful_journeys'] = num_successful
    print("Added date and previous journey information.")

    df['days_into_journey'] = [(current_datetime - first_datetime) / np.timedelta64(1, 'D') for current_datetime, first_datetime in zip(df['current_datetime'], df['start_date'])]
    print("Added number of days since the journey began.")

    df['last_event'] = df['events'].apply(lambda x: x[-1])
    print("Added last event in journey")

    df["days_since_last_event"] = [
        (cur - ts[-1]) / np.timedelta64(1, 'D') for ts, cur in zip(df["timestamps"], df["current_datetime"].dt.tz_localize(None))
    ]
    print("Added number of days since last event.")

    n = len(df)
    result = np.empty(n, dtype=np.float32)  # float32 saves memory
    result[:] = np.nan

    current_dt = df['current_datetime'].dt.tz_localize(None).values

    for i, (events, timestamps) in enumerate(zip(df['events'], df['timestamps'])):
        
        # scan backwards to find last valid event
        for j in range(len(events) - 1, -1, -1):
            if events[j] not in (1, 20, 21):
                delta = current_dt[i] - timestamps[j]
                result[i] = delta / np.timedelta64(1, 'D')
                break

    df['days_since_last_user_event'] = result
    print("Added days since last user action")

    quantiles = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1]
    q_cols = [f"days_between_actions_q_{int(q*100)}" for q in quantiles]
    def compute_row_quantiles(ts):
        ts = np.asarray(ts)
        if ts.size < 2:
            return [np.nan] * len(quantiles)

        diffs_days = np.diff(ts) / np.timedelta64(1, 'D')
        return np.quantile(diffs_days, quantiles)

    result = np.vstack([
        compute_row_quantiles(ts)
        for ts in df['timestamps']
    ])
    df[q_cols] = result
    print("Added quantiles for the number of days between actions in the journey.")

    df['first_event'] = [events[0] for events in df['events']]
    print("Added first event id in journey.")
    
    repeatable_events = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 27, 29]
    event_counts = df['events'].apply(Counter)
    counts_df = (
        pd.DataFrame(event_counts.tolist())
        .reindex(columns=repeatable_events)
        .fillna(0)
        .astype(int)
        .rename(columns=lambda e: f'event_{e}_count')
    )

    df = df.join(counts_df)
    print("Added number of occurrences for repeatable events.")

    milestone_to_events = {
        1: (12, 15),
        2: (7, 18),
        3: (29,),
        4: (8,),
        5: (27,)
    }
    for milestone, events in milestone_to_events.items():
        df[f'reached_milestone_{milestone}'] = [any([event in journey_events for event in events]) for journey_events in df['events']]
    print("Added indicator for whether each milestone was reached.")

    # Add the number of days that it takes to reach each milestone in the journey (assuming that the milestone was actually achieved)
    for milestone in range(1, 6):
        df[f'days_to_milestone_{milestone}'] = [(timestamps[np.where(np.isin(events, milestone_to_events[milestone]))[0][0]] - timestamps[0]) / np.timedelta64(1, 'D') if indicator else np.nan for timestamps, events, indicator in zip(df['timestamps'], df['events'], df[f'reached_milestone_{milestone}'])]
    print("Added the number of days until each milestone was achieved during the journey.")

    df.to_parquet(f'data/{output_file}.parquet', index=False)
    print("Successfully created parquet file.")
