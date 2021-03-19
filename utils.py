def count_files(folder):
    '''Counts all files in subdirectories a give directory

    Parameters:
    -----------
    folder: The directory to containing the files to count

    Returns:
    --------
    count:  int. The number of files not include folders, but incuding all
    subdirectories in the given path.
    '''

    import os

    # get subdirectories
    paths = [path for path, subdirs, files in os.walk(folder) if path != folder]
    num = 0

    # recurse subdirectories and count files
    for path in paths:
        num += len(os.listdir(path))
    return num

def log_runs(runs, net_type='mlp'):
    '''Adds the new runs to the the log files

    Parameters
    ----------
    runs: a list of azure.core.Run objects

    net_type: an abreviation that is used to identify the neural network type
    and as suffix to the log file for keeping logs separate

    Returns
    -------
    DataFrame of log file
    '''

    import pandas as pd
    import os
    import pickle

    runs_dict = {'id':[],
                 'name':[],
                 'type':[],
                 'runtime':[],
                 'test_loss':[],
                 'test_accuracy':[],
                 'train_loss':[],
                 'train_accuracy':[],
                 'val_loss':[],
                 'val_accuracy':[]}

    # loop through runs and fill data
    for run in runs:

        runs_dict['id'].append(run.id)
        runs_dict['name'].append(run.properties['name'])
        runs_dict['type'].append(net_type)
        runs_dict['runtime'].append(get_runtime(run))

        # fetch run metrics from Azure
        metrics = run.get_metrics()

        # this takes a while so give some feedback
        display(metrics)

        runs_dict['test_loss'].append(metrics['test_loss'])
        runs_dict['test_accuracy'].append(metrics['test_accuracy'])

        # if the history file is not alread downloaded, download all files
        if not os.path.exists(os.path.join('models/',f'{run.id}.history')):
            run.download_files(prefix='outputs/',
                               output_directory='models/',
                               append_prefix=False)

        # history was pickled on server, load it
        with open(f'models/{run.id}.history', 'rb') as f:
            history = pickle.load(f)

        runs_dict['train_loss'].append(history['loss'])
        runs_dict['train_accuracy'].append(history['acc'])
        runs_dict['val_loss'].append(history['val_loss'])
        runs_dict['val_accuracy'].append(history['val_acc'])

    # convert to DataFrame
    df = pd.DataFrame(runs_dict)
    df = df.set_index('id')

    # If the log file exists fill the DataFrame with it
    try:
        df_prev = pd.read_pickle('logs/model_log_'+net_type)
    except:
        df_prev = pd.DataFrame([], columns=df.columns)

    # combine previous log with this update
    df = pd.concat([df_prev,df])

    # remove duplicates
    df = df[~df.index.duplicated()]

    # store the log
    df.to_pickle('logs/model_log_'+net_type)
    return df

def visualize_log(log):
    '''Plots the Training and Validation Accuracy and Loss for each row of the
    log

    Parameters
    ----------
    log: DataFrame created by log_runs'''

    import matplotlib.pyplot as plt

    # for each row in the log DataFrame plot accuracy and loss
    for key in log.index:
        row = log.loc[key]
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
        ax1.plot(row['train_accuracy'], label='Train Accuracy')
        ax1.plot(row['val_accuracy'], label='Validation Accuracy')
        ax2.plot(row['train_loss'], label='Train Loss')
        ax2.plot(row['val_loss'], label='Validation Loss')

        ax1.set_title(f'{row["name"]} accuracy\n({key})')
        ax1.legend()
        ax2.set_title(f'{row["name"]} loss\n({key})')
        ax2.legend()
        plt.show()

def plot_accuracy(log):
    '''Plots the test_accuracy for all rows of the log

        Parameters
        ----------
        log: DataFrame created by log_runs'''

    import matplotlib.pyplot as plt

    # plot the test accuracy and loss for all models in the log
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
    log[['test_accuracy','name']].set_index('name').plot.bar(ax=ax1,
                                                             legend=False)
    ax1.set_title('Test Accuracy by model name')
    ax1.set_xlabel('Model Name')

    log[['test_loss','name']].set_index('name').plot.bar(ax=ax2,
                                                         legend=False)
    ax2.set_title('Test Loss by model name')
    ax2.set_xlabel('Model Name')

def get_runtime(run):
    '''Calculates the duration of a azure.core.Run object

    Paramters:
    ----------
    run: azure.core.Run object (should be a completed run)

    Returns:
    --------
    datetime time delta indicating the total run time from start to end
    '''
    
    from datetime import datetime

    # fetch the run details
    details = run.get_details()

    # get start and end time
    start = datetime.strptime(details['startTimeUtc'], '%Y-%m-%dT%H:%M:%S.%fZ')
    end = datetime.strptime(details['endTimeUtc'], '%Y-%m-%dT%H:%M:%S.%fZ')
    return end - start
